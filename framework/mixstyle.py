"""MixStyle domain generalisation layer (Zhou et al. 2021).

Reference: "Domain Generalization with MixStyle", ICLR 2021.

MixStyle randomly mixes the instance-normalisation statistics (mean, std) of
feature maps from different samples during training.  This simulates plausible
new domains by interpolating recording-condition statistics, forcing the model
to rely on domain-invariant features (mosquito wingbeat frequency pattern)
rather than domain-specific cues (recording setup noise floor, microphone
response).

Insertion points in this codebase:
  MTRCNNClassifier — after ConvStage 0 of every branch  (shared instance)
  ASTClassifier   — after patch_embed, before sequence flatten

Both locations are the earliest point where meaningful spatial feature
statistics exist.  A single shared MixStyle instance is used so that the
same (lam, perm) is applied consistently across all three MTRCNN branches.

Config keys:
    use_mixstyle   bool   False
    mixstyle_alpha float  0.1    Beta concentration (lower = weaker mixing)
    mixstyle_p     float  0.5    probability of applying per forward call
"""

import random

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """Randomly mix instance-normalisation statistics across samples.

    Applied to 4-D feature maps [B, C, H, W] only during training.
    At inference (``model.eval()``) it is a pure identity.

    Args:
        alpha:  Beta distribution concentration parameter.
                Higher → stronger mixing, lower → closer to identity.
                Recommended range: 0.1 – 0.4.
        p:      probability of applying per forward call.
        eps:    numerical stability constant for variance.
    """

    def __init__(self, alpha: float = 0.1, p: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.p     = p
        self.eps   = eps
        self._beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, H, W]
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        if B < 2:  # need at least two samples to mix
            return x

        # Per-sample instance statistics  [B, C, 1, 1]
        mu  = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()

        # Normalise
        x_norm = (x - mu) / sig

        # Sample mixing coefficient  [B, 1, 1, 1]
        lam  = self._beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)

        # Interpolate statistics
        mu_mix  = lam * mu  + (1.0 - lam) * mu[perm]
        sig_mix = lam * sig + (1.0 - lam) * sig[perm]

        return x_norm * sig_mix + mu_mix

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, p={self.p}"
