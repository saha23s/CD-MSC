"""PerchClassifier: linear probe on top of frozen Perch v2 embeddings.

Perch v2 (bird-vocalization-classifier/4 on TF Hub) produces one 1280-dim
embedding per 5-second audio window. The extraction step is handled by
extract_perch_features.py; this class sees pre-computed [B, T_windows, 1280]
tensors and outputs the same dict interface as MTRCNNClassifier / ASTClassifier.

Config keys (all optional):
    perch_embed_dim      int    128    projection dim after the 1280-d Perch space
    dropout              float  0.1    dropout before classification heads
    domain_adversarial   bool   False  enable GRL for DANN
    grl_lambda_max       float  1.0    max GRL coefficient
    contrastive_proj_dim int    0      >0 adds a 2-layer projection head for
                                       DiCL / SCoL contrastive losses

Smoke test::

    if __name__ == "__main__":
        cfg = {}
        model = PerchClassifier(cfg, num_species_classes=9, num_domain_classes=5)
        feats   = torch.randn(4, 3, 1280)
        lengths = torch.tensor([3, 2, 1, 3])
        out = model(feats, lengths)
        assert out["species_logits"].shape == (4, 9)
        assert out["domain_logits"].shape  == (4, 5)
        print("PerchClassifier OK — params:",
              sum(p.numel() for p in model.parameters()))

Author: Sulagna Saha
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from framework.gradient_reversal import GradientReversalLayer


class PerchClassifier(nn.Module):
    """Linear probe on frozen Perch v2 embeddings.

    Args:
        features: [B, T_windows, 1280]  pre-extracted Perch embeddings
        lengths:  [B]  number of valid windows per sample (≥ 1)

    Returns:
        dict with ``species_logits``, ``domain_logits``, ``embedding``,
        and optionally ``proj_embedding``.
    """

    PERCH_DIM = 1_280

    def __init__(self, config: dict, num_species_classes: int, num_domain_classes: int) -> None:
        super().__init__()

        embed_dim = int(config.get("perch_embed_dim", config.get("embed_dim", 128)))
        dropout   = float(config.get("dropout", 0.1))

        # Normalise raw Perch embeddings before projecting.
        self.input_norm = nn.LayerNorm(self.PERCH_DIM)
        self.projection = nn.Sequential(
            nn.Linear(self.PERCH_DIM, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.species_classifier = nn.Linear(embed_dim, num_species_classes)
        self.domain_classifier  = nn.Linear(embed_dim, num_domain_classes)

        self.grl: Optional[GradientReversalLayer] = (
            GradientReversalLayer(lambda_=0.0) if config.get("domain_adversarial", False) else None
        )

        proj_dim: int = config.get("contrastive_proj_dim", 0)
        self.contrastive_proj: Optional[nn.Sequential] = (
            nn.Sequential(
                nn.Linear(embed_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, proj_dim)
            )
            if proj_dim > 0 else None
        )

    def set_grl_lambda(self, lambda_: float) -> None:
        if self.grl is not None:
            self.grl.set_lambda(lambda_)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = features.shape

        # Masked mean-pool over valid Perch windows.  [B, T, 1280] → [B, 1280]
        valid_lengths = lengths.clamp(min=1, max=T)
        time_idx = torch.arange(T, device=features.device).unsqueeze(0)   # [1, T]
        mask = time_idx < valid_lengths.unsqueeze(1)                       # [B, T]
        pooled = (features * mask.unsqueeze(2).float()).sum(dim=1) / valid_lengths.unsqueeze(1).float()

        embedding    = self.projection(self.input_norm(pooled))
        domain_input = self.grl(embedding) if self.grl is not None else embedding

        out: Dict[str, torch.Tensor] = {
            "species_logits": self.species_classifier(embedding),
            "domain_logits":  self.domain_classifier(domain_input),
            "embedding":      embedding,
        }
        if self.contrastive_proj is not None:
            out["proj_embedding"] = self.contrastive_proj(embedding)
        return out


if __name__ == "__main__":
    cfg = {}
    model = PerchClassifier(cfg, num_species_classes=9, num_domain_classes=5)
    feats   = torch.randn(4, 3, 1280)
    lengths = torch.tensor([3, 2, 1, 3])
    out = model(feats, lengths)
    assert out["species_logits"].shape == (4, 9), out["species_logits"].shape
    assert out["domain_logits"].shape  == (4, 5), out["domain_logits"].shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PerchClassifier OK — trainable params: {n_params:,}")
