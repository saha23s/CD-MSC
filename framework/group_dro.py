"""GroupDRO: Group Distributionally Robust Optimization (Sagawa et al., 2020).

Maintains per-domain loss weights q that are upweighted exponentially when a
domain's loss is high and downweighted otherwise. At each batch the species
classification loss is replaced by the q-weighted sum of per-domain losses,
biasing the model toward improving the worst-performing domain.

Reference: https://arxiv.org/abs/1911.08731
"""

import torch
import torch.nn.functional as F


class GroupDROState:
    """Persistent per-domain weight vector updated once per batch.

    Args:
        num_domains: number of training domains.
        eta:         step size for exponential weight update (0.01–0.1 typical).
    """

    def __init__(self, num_domains: int, eta: float = 0.01):
        self.q   = torch.ones(num_domains) / num_domains  # uniform init
        self.eta = eta

    def to(self, device: torch.device) -> "GroupDROState":
        self.q = self.q.to(device)
        return self

    def weighted_loss(
        self,
        species_logits: torch.Tensor,
        species_labels: torch.Tensor,
        domain_labels:  torch.Tensor,
    ) -> torch.Tensor:
        """Compute q-weighted species CE loss and update q in-place.

        For each domain present in the batch, computes the mean CE loss,
        updates q_d ∝ q_d * exp(η * L_d), renormalises, then returns the
        weighted sum Σ_d q_d * L_d.

        Domains absent from the batch keep their q unchanged.
        Returns standard mean CE loss if no domain has >0 samples (safe fallback).
        """
        unique_domains = domain_labels.unique()
        per_domain_loss = {}

        for d in unique_domains:
            mask = domain_labels == d
            if mask.sum() == 0:
                continue
            per_domain_loss[d.item()] = F.cross_entropy(
                species_logits[mask], species_labels[mask]
            )

        if not per_domain_loss:
            return F.cross_entropy(species_logits, species_labels)

        # Exponential weight update (detached — no gradient through q)
        for d_idx, loss_d in per_domain_loss.items():
            self.q[d_idx] = self.q[d_idx] * torch.exp(self.eta * loss_d.detach())
        self.q = self.q / self.q.sum()

        # Weighted loss (gradients flow through per_domain_loss, not q)
        weighted = sum(
            self.q[d_idx] * loss_d for d_idx, loss_d in per_domain_loss.items()
        )
        return weighted
