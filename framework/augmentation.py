"""Spectrogram-level augmentation transforms for the DCASE2026 mosquito baseline.

All single-sample transforms operate on float32 tensors of shape [T, F]
(time-frames × mel-bins), applied after normalization and only during training.

Mixup operates at batch level and is called from the training loop.
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import torch


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
