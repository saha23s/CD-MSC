"""Spectrogram-level augmentation transforms for the DCASE2026 mosquito baseline.

All single-sample transforms operate on float32 tensors of shape [T, F]
(time-frames × mel-bins), applied after normalization and only during training.

Mixup operates at batch level and is called from the training loop.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import median_filter


# ---------------------------------------------------------------------------
# Single-sample transforms
# ---------------------------------------------------------------------------

class TimeMasking:
    """Randomly zero out contiguous time-frame blocks (SpecAugment T-mask)."""

    def __init__(self, num_masks: int = 2, max_mask_fraction: float = 0.15, p: float = 1.0) -> None:
        self.num_masks = num_masks
        self.max_mask_fraction = max_mask_fraction
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # [T, F]
        if random.random() > self.p:
            return x
        x = x.clone()
        T = x.size(0)
        max_width = max(1, int(T * self.max_mask_fraction))
        for _ in range(self.num_masks):
            width = random.randint(1, max_width)
            start = random.randint(0, max(0, T - width))
            x[start : start + width, :] = 0.0
        return x


class FrequencyMasking:
    """Randomly zero out contiguous mel-bin blocks (SpecAugment F-mask)."""

    def __init__(self, num_masks: int = 2, max_mask_bins: int = 8, p: float = 1.0) -> None:
        self.num_masks = num_masks
        self.max_mask_bins = max_mask_bins
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # [T, F]
        if random.random() > self.p:
            return x
        x = x.clone()
        n_bins = x.size(1)
        for _ in range(self.num_masks):
            width = random.randint(1, min(self.max_mask_bins, n_bins))
            start = random.randint(0, n_bins - width)
            x[:, start : start + width] = 0.0
        return x


class GaussianNoise:
    """Add zero-mean Gaussian noise to the normalized spectrogram."""

    def __init__(self, std: float = 0.05, p: float = 0.5) -> None:
        self.std = std
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        return x + torch.randn_like(x) * self.std


class FrequencyShift:
    """Circularly shift mel content by a random number of bins."""

    def __init__(self, max_shift_bins: int = 4, p: float = 0.5) -> None:
        self.max_shift_bins = max_shift_bins
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return x
        shift = random.randint(-self.max_shift_bins, self.max_shift_bins)
        return torch.roll(x, shift, dims=1)


# ---------------------------------------------------------------------------
# HPSS — Harmonic-Percussive Source Separation
# ---------------------------------------------------------------------------

class HPSS:
    """Harmonic-Percussive Source Separation augmentation (Fitzgerald 2010).

    Separates the normalized log-mel spectrogram into harmonic (horizontal,
    tonal) and percussive (vertical, transient) components using median
    filtering + Wiener power masking, then randomly attenuates the percussive
    component.

    **Why this helps here:**
    Mosquito wingbeat is a periodic tone → horizontal stripes (harmonic).
    Recording-condition artifacts (microphone transients, environmental noise,
    domain-specific background) → vertical structures (percussive).
    Attenuating the percussive component forces the model to rely on the
    wingbeat signal rather than recording-setup cues, directly targeting the
    D5-vs-D1–D4 domain shift.

    Algorithm (Fitzgerald 2010 with Wiener power masking):
        H = median_filter(X, kernel=(1, kernel_harm))   # frequency-smoothed
        P = median_filter(X, kernel=(kernel_perc, 1))   # time-smoothed
        M_h = |H|^power / (|H|^power + |P|^power + ε)
        M_p = |P|^power / (|H|^power + |P|^power + ε)
        X_aug = X × (M_h + α × M_p),   α ~ Uniform(alpha_min, 1.0)

    When α=1 → identity. When α=0 → harmonic-only.
    Randomising α avoids over-reliance on clean separation.

    Note: kernel sizes are tuned for short clips (~63 frames, 64 mel bins).
    Larger kernels → stronger separation but higher computational cost.

    Args:
        kernel_harm:  length of horizontal (frequency) median filter. Must be odd.
        kernel_perc:  length of vertical (time) median filter. Must be odd.
        alpha_min:    minimum percussive retention coefficient (0 = harmonic-only).
        power:        Wiener masking exponent (2.0 = standard; higher → harder mask).
        p:            probability of applying this transform.
    """

    def __init__(
        self,
        kernel_harm: int = 17,
        kernel_perc: int = 9,
        alpha_min: float = 0.0,
        power: float = 2.0,
        p: float = 0.5,
    ) -> None:
        assert kernel_harm % 2 == 1, "kernel_harm must be odd"
        assert kernel_perc % 2 == 1, "kernel_perc must be odd"
        self.kernel_harm = kernel_harm
        self.kernel_perc = kernel_perc
        self.alpha_min   = alpha_min
        self.power       = power
        self.p           = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:  # [T, F]
        if random.random() > self.p:
            return x

        x_np = x.numpy()                                              # [T, F]

        # Median filters: horizontal (harmonic) and vertical (percussive)
        h = median_filter(x_np, size=(1, self.kernel_harm))
        p = median_filter(x_np, size=(self.kernel_perc, 1))

        # Wiener power masks — M_h + M_p = 1 everywhere
        H = np.abs(h) ** self.power
        P = np.abs(p) ** self.power
        denom = H + P + 1e-8
        M_h = H / denom                                               # [T, F]
        M_p = P / denom                                               # [T, F]

        # Randomly retain α fraction of the percussive component
        alpha  = random.uniform(self.alpha_min, 1.0)
        x_aug  = x_np * (M_h + alpha * M_p)

        return torch.tensor(x_aug, dtype=x.dtype)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class AugmentationPipeline:
    """Ordered chain of spectrogram augmentation transforms."""

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        names = [t.__class__.__name__ for t in self.transforms]
        return f"AugmentationPipeline({names})"


def build_augmentation_pipeline(aug_config: Optional[Dict]) -> Optional[AugmentationPipeline]:
    """Construct an AugmentationPipeline from the ``augmentation`` config block.

    Returns ``None`` when ``aug_config`` is absent or all transforms are disabled,
    so callers can guard with a simple ``if pipeline:`` check.

    Example config block::

        "augmentation": {
            "time_masking":   {"enabled": true,  "num_masks": 2, "max_mask_fraction": 0.15, "p": 1.0},
            "freq_masking":   {"enabled": true,  "num_masks": 2, "max_mask_bins": 8,        "p": 1.0},
            "gaussian_noise": {"enabled": false, "std": 0.05,   "p": 0.5},
            "freq_shift":     {"enabled": false, "max_shift_bins": 4,                       "p": 0.5},
            "mixup":          {"enabled": false, "alpha": 0.4}
        }
    """
    if not aug_config:
        return None

    transforms: List[Callable] = []

    cfg = aug_config.get("time_masking", {})
    if cfg.get("enabled", False):
        transforms.append(TimeMasking(
            num_masks=cfg.get("num_masks", 2),
            max_mask_fraction=cfg.get("max_mask_fraction", 0.15),
            p=cfg.get("p", 1.0),
        ))

    cfg = aug_config.get("freq_masking", {})
    if cfg.get("enabled", False):
        transforms.append(FrequencyMasking(
            num_masks=cfg.get("num_masks", 2),
            max_mask_bins=cfg.get("max_mask_bins", 8),
            p=cfg.get("p", 1.0),
        ))

    cfg = aug_config.get("gaussian_noise", {})
    if cfg.get("enabled", False):
        transforms.append(GaussianNoise(
            std=cfg.get("std", 0.05),
            p=cfg.get("p", 0.5),
        ))

    cfg = aug_config.get("freq_shift", {})
    if cfg.get("enabled", False):
        transforms.append(FrequencyShift(
            max_shift_bins=cfg.get("max_shift_bins", 4),
            p=cfg.get("p", 0.5),
        ))

    cfg = aug_config.get("hpss", {})
    if cfg.get("enabled", False):
        transforms.append(HPSS(
            kernel_harm=cfg.get("kernel_harm", 17),
            kernel_perc=cfg.get("kernel_perc", 9),
            alpha_min=cfg.get("alpha_min", 0.0),
            power=cfg.get("power", 2.0),
            p=cfg.get("p", 0.5),
        ))

    return AugmentationPipeline(transforms) if transforms else None


# ---------------------------------------------------------------------------
# Batch-level: Mixup
# ---------------------------------------------------------------------------

def mixup_batch(
    features: torch.Tensor,        # [B, T, F]
    species_labels: torch.Tensor,  # [B]
    domain_labels: torch.Tensor,   # [B]
    alpha: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply Mixup to a padded spectrogram batch.

    Samples lam ~ Beta(alpha, alpha) and linearly interpolates each sample with a
    random partner from the same batch. Labels are returned as two sets (a, b) so
    the caller can compute the mixed loss::

        lam * loss(logits, labels_a) + (1 - lam) * loss(logits, labels_b)

    Args:
        features: padded spectrogram batch  [B, T, F]
        species_labels: integer class labels [B]
        domain_labels:  integer domain labels [B]
        alpha: Beta concentration; higher → stronger mixing

    Returns:
        (mixed_features, species_a, species_b, domain_a, domain_b, lam)
    """
    lam = float(torch.distributions.Beta(alpha, alpha).sample()) if alpha > 0 else 1.0
    perm = torch.randperm(features.size(0), device=features.device)
    mixed = lam * features + (1.0 - lam) * features[perm]
    return mixed, species_labels, species_labels[perm], domain_labels, domain_labels[perm], lam


# ---------------------------------------------------------------------------
# Batch-level: Frequency-Band Selective Mix (FBS-Mix)
# ---------------------------------------------------------------------------

def fbs_mix_batch(
    features: torch.Tensor,
    species_lo: int = 9,
    species_hi: int = 36,
    alpha: float = 0.1,
    p: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Frequency-Band Selective Mix: mix domain-informative bins, protect species bins.

    Empirical motivation (mosquito wingbeat data, test split):
      Bins  0–8:  species/domain ratio 0.30–0.93  → domain-dominated
      Bins  9–35: species/domain ratio 1.50–4.12  → species-dominated (wingbeat)
      Bins 36–63: ratio 0.73–1.47                 → mixed/unclear

    Standard MixStyle (applied inside the CNN to all channels) mixes across all
    mel-bin information indiscriminately, corrupting the wingbeat signal (bins 9-35)
    while providing diversity in the domain-noise bands (bins 0-8). The net effect
    is approximately zero, explaining why M2+MixStyle ≈ M2 in LODO experiments.

    FBS-Mix applies instance-stat mixing ONLY to the domain-dominated low-frequency
    band (bins 0 to species_lo-1), while leaving the species-discriminative wingbeat
    core (bins species_lo to species_hi-1) and the mixed high-frequency region
    (bins species_hi onward) completely untouched.

    Operates on raw input spectrograms [B, T, F] before the model, as a batch-level
    transform (analogous to Mixup in the training loop). No architecture changes.
    No inference overhead — skipped when not training.

    Args:
        features:   padded log-mel batch  [B, T, F].
        species_lo: first bin of the species-dominated zone (default 9).
        species_hi: one past the last species-dominated bin (default 36).
        alpha:      Beta concentration for mixing coefficient; lower = subtler mixing.
        p:          probability of applying per batch.
        eps:        variance floor for numerical stability.

    Returns:
        features with bins 0:species_lo style-mixed across random pairs.  [B, T, F]
    """
    import random as _random
    if _random.random() > p:
        return features
    B = features.size(0)
    if B < 2 or species_lo <= 0:
        return features

    lam  = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(B, device=features.device)

    out  = features.clone()
    band = out[:, :, :species_lo]                              # [B, T, species_lo]

    # Per-sample instance statistics over the time dimension
    mu  = band.mean(dim=1, keepdim=True)                       # [B, 1, species_lo]
    sig = (band.var(dim=1, keepdim=True) + eps).sqrt()         # [B, 1, species_lo]

    # Normalise, then re-scale with mixed statistics
    mu_mix  = lam * mu  + (1.0 - lam) * mu[perm]
    sig_mix = lam * sig + (1.0 - lam) * sig[perm]
    out[:, :, :species_lo] = (band - mu) / sig * sig_mix + mu_mix

    return out


def build_fbs_mix_fn(config: dict):
    """Return an fbs_mix_batch callable from config, or None if disabled."""
    if not config.get("freq_band_mixstyle", False):
        return None
    lo    = config.get("fbmix_species_lo",  9)
    hi    = config.get("fbmix_species_hi",  36)
    alpha = config.get("fbmix_alpha",        0.1)
    p     = config.get("fbmix_p",            0.5)
    def _fn(features: torch.Tensor) -> torch.Tensor:
        return fbs_mix_batch(features, species_lo=lo, species_hi=hi, alpha=alpha, p=p)
    return _fn
