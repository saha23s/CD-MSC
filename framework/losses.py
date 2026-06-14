"""Domain-invariant contrastive and distribution alignment losses.

Implements two loss terms from DR-BioL (Hou et al., 2025, arXiv:2510.00346),
adapted for the severely imbalanced LODO setting (D5 = 99.4% of training data).

DicL — Domain-Invariant Contrastive Loss
-----------------------------------------
Positive pairs are (same species, different domain). This directly pulls
same-species embeddings from different domains together, enforcing that
species identity is domain-agnostic. Complements DANN (which pushes domain
information out of embeddings adversarially) with an explicit attractive
gradient — the key missing signal in our LODO setting.

IMPORTANT: DicL requires cross-domain positive pairs in each batch.
Always combine with domain_balanced_sampling=True.

SdaL — Species-conditional Distribution Alignment via MMD
----------------------------------------------------------
Per-species MMD between cross-domain embedding distributions. Avoids
conflating species-discriminative and domain-discriminative variance
by aligning marginals within each species class rather than globally.
Noisier than DicL with very small per-domain sample counts; use lower weight.

Both losses operate on L2-normalised embeddings from the model's shared
representation layer (MTRCNN: 32-dim; AST: embed_dim-dim CLS token).
"""

import torch
import torch.nn.functional as F

# Genus index for each species (0=Aedes, 1=Culex, 2=Anopheles).
# Indices match SPECIES_NAMES order in framework/metadata.py.
#   0 Ae.aegypti, 1 Ae.albopictus → Aedes (0)
#   2 Cx.quinquefasciatus, 6 Cx.pipiens → Culex (1)
#   3 An.gambiae, 4 An.arabiensis, 5 An.dirus, 7 An.minimus, 8 An.stephensi → Anopheles (2)
_SPECIES_TO_GENUS = torch.tensor([0, 0, 1, 2, 2, 2, 1, 2, 2], dtype=torch.long)


def species_cohesion_contrastive_loss(
    embeddings: torch.Tensor,
    species_labels: torch.Tensor,
    tau: float = 0.01,
) -> torch.Tensor:
    """Species-Cohesion Contrastive Loss (ScoL).

    Positive pairs: same species label, any domain.
    Denominator: all pairs excluding self.

    Implements Eq. (4-5) from DR-BioL (Hou et al., 2025, arXiv:2510.00346).
    Enforces intra-class compactness and inter-class separation in the species
    embedding space. Complements DicL (which requires cross-domain positives)
    with a denser gradient signal from within-domain same-species pairs.

    Args:
        embeddings:     [B, D] model embeddings (any scale; normalised internally).
        species_labels: [B]    integer species indices.
        tau:            Temperature. Paper default 0.01.

    Returns:
        Scalar loss. Returns 0 if no same-species pairs exist in batch.
    """
    z = F.normalize(embeddings, dim=-1)          # [B, D]
    sim = (z @ z.T) / tau                        # [B, B]

    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)

    # Positive mask: same species, excluding self
    same_sp  = species_labels.unsqueeze(0) == species_labels.unsqueeze(1)  # [B, B]
    pos_mask = same_sp & ~eye                                                # [B, B]

    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return embeddings.new_tensor(0.0)

    log_denom = torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1)  # [B]

    n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
    loss_per_anchor = -(sim * pos_mask).sum(dim=1) / n_pos + log_denom       # [B]

    return loss_per_anchor[has_pos].mean()


def domain_invariant_contrastive_loss(
    embeddings: torch.Tensor,
    species_labels: torch.Tensor,
    domain_labels: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Domain-Invariant Contrastive Loss (DicL).

    Positive pairs: same species, different domain.
    Denominator: all pairs excluding self.

    Implements Eq. (3) from DR-BioL with cosine similarity (L2-normed dot product)
    instead of raw dot product for numerical stability.

    Args:
        embeddings:     [B, D] model embeddings (any scale; normalised internally).
        species_labels: [B]    integer species indices.
        domain_labels:  [B]    integer domain indices.
        tau:            Temperature. Lower = sharper distribution over negatives.
                        Default 0.07 (vs 0.01 in paper — more stable with small batches).

    Returns:
        Scalar loss. Returns 0 if no cross-domain positive pairs exist in batch
        (e.g. single-domain batch despite balanced sampling).
    """
    z = F.normalize(embeddings, dim=-1)          # [B, D]
    sim = (z @ z.T) / tau                        # [B, B]

    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)

    # Positive mask: same species AND different domain (excluding self)
    same_sp  = species_labels.unsqueeze(0) == species_labels.unsqueeze(1)   # [B, B]
    diff_dom = domain_labels.unsqueeze(0)  != domain_labels.unsqueeze(1)    # [B, B]
    pos_mask = same_sp & diff_dom & ~eye                                     # [B, B]

    # Anchors with at least one positive
    has_pos = pos_mask.any(dim=1)                                            # [B]
    if not has_pos.any():
        return embeddings.new_tensor(0.0)

    # Log-sum-exp denominator over all non-self pairs
    log_denom = torch.logsumexp(sim.masked_fill(eye, float("-inf")), dim=1)  # [B]

    # Mean log-prob over positives for each anchor
    n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
    loss_per_anchor = -(sim * pos_mask).sum(dim=1) / n_pos + log_denom       # [B]

    return loss_per_anchor[has_pos].mean()


def _rbf_mmd2(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Unbiased RBF-kernel MMD² between two sample sets.

    Args:
        x: [n, D] samples from distribution P.
        y: [m, D] samples from distribution Q.
        sigma: RBF bandwidth (applied to squared Euclidean distances).

    Returns:
        Scalar MMD² estimate.
    """
    def rbf(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Pairwise squared distances via expansion
        sq = (a.unsqueeze(1) - b.unsqueeze(0)).pow(2).sum(-1)   # [|a|, |b|]
        return torch.exp(-sq / (2 * sigma ** 2))

    k_xx = rbf(x, x)
    k_yy = rbf(y, y)
    k_xy = rbf(x, y)

    n, m = x.size(0), y.size(0)
    # Unbiased: subtract diagonal for within-set terms
    mmd2 = (
        (k_xx.sum() - k_xx.trace()) / max(n * (n - 1), 1)
        + (k_yy.sum() - k_yy.trace()) / max(m * (m - 1), 1)
        - 2 * k_xy.mean()
    )
    return mmd2


def species_conditional_mmd_loss(
    embeddings: torch.Tensor,
    species_labels: torch.Tensor,
    domain_labels: torch.Tensor,
    sigma: float = 1.0,
    min_samples: int = 2,
) -> torch.Tensor:
    """Species-conditional Distribution Alignment via MMD (SdaL).

    For each species present in the batch, computes MMD² between all pairs of
    domain-specific sub-distributions and averages. Aligns per-species marginals
    rather than global marginals, preventing alignment from conflating class and
    domain variance.

    Args:
        embeddings:     [B, D] model embeddings (normalised internally).
        species_labels: [B]    integer species indices.
        domain_labels:  [B]    integer domain indices.
        sigma:          RBF kernel bandwidth.
        min_samples:    Skip a species-domain subset if it has fewer samples
                        (kernel estimate too noisy). Default 2.

    Returns:
        Scalar loss (mean MMD² across valid species-domain pairs).
        Returns 0 if no valid pairs found.
    """
    z = F.normalize(embeddings, dim=-1)
    unique_species = species_labels.unique()
    unique_domains  = domain_labels.unique()

    mmd_terms = []
    for sp in unique_species:
        sp_mask = species_labels == sp
        # Collect per-domain embeddings for this species
        domain_subsets = []
        for dom in unique_domains:
            mask = sp_mask & (domain_labels == dom)
            if mask.sum() >= min_samples:
                domain_subsets.append(z[mask])

        if len(domain_subsets) < 2:
            continue
        # All cross-domain pairs for this species
        for i in range(len(domain_subsets)):
            for j in range(i + 1, len(domain_subsets)):
                mmd_terms.append(_rbf_mmd2(domain_subsets[i], domain_subsets[j], sigma))

    if not mmd_terms:
        return embeddings.new_tensor(0.0)
    return torch.stack(mmd_terms).mean()


def genus_hard_negative_contrastive_loss(
    embeddings: torch.Tensor,
    species_labels: torch.Tensor,
    tau: float = 0.05,
    genus_neg_scale: float = 5.0,
) -> torch.Tensor:
    """Genus-aware contrastive loss.

    Addresses Anopheles genus collapse: An.gambiae, An.arabiensis, and An.dirus
    only appear in one training domain each (D5/D4), so DicL produces no
    cross-domain positive pairs for them. This loss enforces within-batch
    intra-genus separation without requiring cross-domain pairs.

    Positive pairs: same species (any domain) — same as ScoL.
    Hard negatives: same genus, different species — their similarity score
    is boosted in the denominator by ``genus_neg_scale`` (log-space additive)
    so the model is explicitly penalised for collapsing intra-genus distinctions.

    Args:
        embeddings:      [B, D] model embeddings (normalised internally).
        species_labels:  [B]    integer species indices (0–8).
        tau:             Temperature. Default 0.05 (tighter than ScoL 0.01 to
                         avoid over-confident gradients with hard negatives).
        genus_neg_scale: Multiplier on same-genus different-species similarity
                         in the denominator (log-space: adds log(scale)).
                         Higher = stronger push away from genus confusers.

    Returns:
        Scalar loss. 0 if no same-species pairs in batch.
    """
    z = F.normalize(embeddings, dim=-1)          # [B, D]
    sim = (z @ z.T) / tau                        # [B, B]

    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)

    same_sp = species_labels.unsqueeze(0) == species_labels.unsqueeze(1)   # [B, B]
    pos_mask = same_sp & ~eye

    has_pos = pos_mask.any(dim=1)
    if not has_pos.any():
        return embeddings.new_tensor(0.0)

    # Genus labels derived from species labels
    genus = _SPECIES_TO_GENUS.to(device=z.device)[species_labels]          # [B]
    same_genus = genus.unsqueeze(0) == genus.unsqueeze(1)                   # [B, B]
    hard_neg_mask = same_genus & ~same_sp & ~eye                            # [B, B]

    # Boost same-genus different-species scores in denominator
    log_scale = torch.log(torch.tensor(genus_neg_scale, device=z.device))
    sim_boosted = sim + hard_neg_mask.float() * log_scale                   # [B, B]

    log_denom = torch.logsumexp(sim_boosted.masked_fill(eye, float("-inf")), dim=1)  # [B]

    n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
    loss_per_anchor = -(sim * pos_mask).sum(dim=1) / n_pos + log_denom     # [B]

    return loss_per_anchor[has_pos].mean()
