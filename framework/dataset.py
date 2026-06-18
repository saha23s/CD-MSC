"""Dataset helpers for reading precomputed feature files.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import json
import pickle
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset


def clip_instance_normalize(feature: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Normalize each mel bin by its own mean and std across the clip's time frames.

    Removes recording-level spectral coloring (device EQ, gain) while preserving
    temporal dynamics and relative spectral shape within each frame.

    Args:
        feature: [T, F] log-mel spectrogram.
        eps: floor for std to avoid division by zero on silent/short clips.

    Returns:
        Normalized feature of the same shape.
    """
    mean = feature.mean(axis=0, keepdims=True)          # [1, F]
    std  = feature.std(axis=0, keepdims=True)           # [1, F]
    return (feature - mean) / np.maximum(std, eps)


# --- Physics-grounded wingbeat descriptor -----------------------------------
# Fixed-dimensional, amplitude-invariant summary of the species-discriminative
# wingbeat band (≈500–2200 Hz, the bins flagged species-dominated by the
# frequency-band variance analysis on the test split). Computed from the
# *un-normalised* dB log-mel energy so it is a genuine physical quantity rather
# than a z-score artefact relative to the (D5-dominated) training statistics.
WINGBEAT_DESCRIPTOR_DIM = 3
_WB_LO_HZ, _WB_HI_HZ = 500.0, 2200.0


def wingbeat_band_bins(
    n_mels: int, fmin: float, fmax: float,
    lo: float = _WB_LO_HZ, hi: float = _WB_HI_HZ,
) -> Tuple[int, int]:
    """Return (start, end) mel-bin indices covering ``lo``–``hi`` Hz.

    Uses the same Slaney mel scale as the feature extractor
    (``torchlibrosa.LogmelFilterBank`` → ``librosa.filters.mel``, ``htk=False``),
    so the band aligns with the actual stored mel bins.
    """
    import librosa

    centers = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=max(fmax, 1.0), htk=False)
    start = int(np.searchsorted(centers, lo))
    end   = int(np.searchsorted(centers, hi))
    return max(0, min(start, n_mels - 1)), min(n_mels, max(end, start + 1))


def wingbeat_params_from_config(config: Dict) -> Optional[Dict]:
    """Wingbeat descriptor parameters from a resolved config, or None if disabled."""
    if not config.get("use_wingbeat_feature", False):
        return None
    return {"n_mels": config["n_mels"], "fmin": config.get("fmin", 0.0), "fmax": config["fmax"]}


def compute_wingbeat_descriptor(feature_db: np.ndarray, start: int, end: int) -> np.ndarray:
    """Energy-weighted spectral summary of the wingbeat band.

    Args:
        feature_db: [T, n_mels] dB log-mel power *before* any normalisation
            (stored features are ``10·log10(mel_power)``, ref=1.0).
        start, end: mel-bin slice [start, end) defining the wingbeat band.

    Returns:
        [3] float32 = (centroid, bandwidth, band_energy_fraction):
          - centroid: energy-weighted mean bin, normalised to [0, 1] across band;
          - bandwidth: std of the band power distribution (harmonic richness);
          - band_energy_fraction: in-band / total energy — a tonality/SNR proxy.

        All three are invariant to a global amplitude scaling. Only frames whose
        in-band energy is ≥ the per-clip median are aggregated, gating out
        silence/noise frames where no insect is in flight.
    """
    band = feature_db[:, start:end]                       # [T, n_band] dB
    if band.size == 0:
        return np.zeros(WINGBEAT_DESCRIPTOR_DIM, dtype=np.float32)
    power_band = np.power(10.0, band / 10.0)              # [T, n_band] linear power
    power_all  = np.power(10.0, feature_db / 10.0)        # [T, n_mels]
    e_t = power_band.sum(axis=1)                          # [T] per-frame in-band energy
    if e_t.sum() <= 0:
        return np.zeros(WINGBEAT_DESCRIPTOR_DIM, dtype=np.float32)

    keep = e_t >= np.median(e_t)                          # energy gate → flight frames
    if not keep.any():
        keep = np.ones_like(e_t, dtype=bool)

    pooled = power_band[keep].sum(axis=0)                 # [n_band] pooled in-band power
    total  = pooled.sum()
    if total <= 0:
        return np.zeros(WINGBEAT_DESCRIPTOR_DIM, dtype=np.float32)

    p   = pooled / total                                  # distribution over band
    idx = np.linspace(0.0, 1.0, p.shape[0], dtype=np.float64)
    centroid  = float((p * idx).sum())
    bandwidth = float(np.sqrt((p * (idx - centroid) ** 2).sum()))
    band_frac = float(total / power_all[keep].sum())
    return np.array([centroid, bandwidth, band_frac], dtype=np.float32)


def load_feature_payload(path: Union[str, Path]) -> Dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_feature_stats(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    return mean, std


def validate_feature_payload(payload: Dict, expected_signature: Optional[str]) -> None:
    if expected_signature is None:
        return
    if payload.get("config_signature") != expected_signature:
        raise ValueError("Feature file does not match the current configuration.")


def validate_feature_stats_payload(path: Union[str, Path], expected_signature: Optional[str]) -> None:
    if expected_signature is None:
        return
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("feature_config_signature") != expected_signature:
        raise ValueError("Feature statistics file does not match the current training feature configuration.")


def compute_domain_stats(items: List[Dict]) -> Dict[str, tuple]:
    """Per-domain (mean, std) over feature frames for per-domain z-score normalisation.

    Aligns each recording domain to a common zero-mean / unit-variance region of
    feature space, removing the per-domain offset that dominates cross-domain shift
    (the D5 cluster sits ~1.5 from every other domain in Perch space). Computed
    transductively from whatever items the dataset holds, so the held-out LODO
    domain is normalised by its own statistics (test-time adaptation).
    """
    sums: Dict[str, np.ndarray] = {}
    sqsums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for item in items:
        feature = item["feature"].astype(np.float64)          # [T, D]
        domain = item["domain"]
        s = feature.sum(axis=0)
        sq = np.square(feature).sum(axis=0)
        if domain not in sums:
            sums[domain], sqsums[domain], counts[domain] = s, sq, feature.shape[0]
        else:
            sums[domain] += s
            sqsums[domain] += sq
            counts[domain] += feature.shape[0]
    stats: Dict[str, tuple] = {}
    for domain in sums:
        mean = (sums[domain] / counts[domain]).astype(np.float32)
        var = np.maximum(sqsums[domain] / counts[domain] - np.square(mean.astype(np.float64)), 1e-12)
        stats[domain] = (mean, np.sqrt(var).astype(np.float32))
    return stats


class MosquitoFeatureDataset(Dataset):
    def __init__(
        self,
        feature_pickle_path: Union[str, Path],
        feature_stats_path: Optional[Union[str, Path]] = None,
        max_train_frames: Optional[int] = None,
        max_eval_frames: Optional[int] = None,
        training: bool = False,
        normalize_features: bool = True,
        clip_normalize: bool = False,
        expected_feature_signature: Optional[str] = None,
        expected_stats_signature: Optional[str] = None,
        augment: Optional[Callable] = None,
        wingbeat_params: Optional[Dict] = None,
        per_domain_norm: bool = False,
    ) -> None:
        payload = load_feature_payload(feature_pickle_path)
        validate_feature_payload(payload, expected_feature_signature)
        self.samples = payload["items"]
        self.training = training
        self.max_train_frames = max_train_frames
        self.max_eval_frames = max_eval_frames
        self.clip_normalize = clip_normalize
        self.feature_mean = None
        self.feature_std = None
        self.augment = augment
        self.wb_bins = wingbeat_band_bins(**wingbeat_params) if wingbeat_params else None
        self.per_domain_norm = per_domain_norm
        self.domain_stats: Optional[Dict[str, tuple]] = None
        if per_domain_norm:
            # Per-domain z-score from this split's own items (transductive).
            self.domain_stats = compute_domain_stats(self.samples)
            self.normalize_features = True
        else:
            self.normalize_features = normalize_features and feature_stats_path is not None
            if self.normalize_features:
                validate_feature_stats_payload(feature_stats_path, expected_stats_signature)
                self.feature_mean, self.feature_std = load_feature_stats(feature_stats_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_crop(self, feature: np.ndarray) -> np.ndarray:
        if self.training and self.max_train_frames and feature.shape[0] > self.max_train_frames:
            start = random.randint(0, feature.shape[0] - self.max_train_frames)
            return feature[start : start + self.max_train_frames]
        if not self.training and self.max_eval_frames and feature.shape[0] > self.max_eval_frames:
            # Centre-crop: preserves the wingbeat-rich middle of long clips
            mid   = feature.shape[0] // 2
            half  = self.max_eval_frames // 2
            start = max(0, mid - half)
            return feature[start : start + self.max_eval_frames]
        return feature

    def _normalize(self, feature: np.ndarray, domain: str) -> np.ndarray:
        if not self.normalize_features:
            return feature
        if self.per_domain_norm:
            mean, std = self.domain_stats[domain]
            return (feature - mean) / np.maximum(std, 1e-8)
        return (feature - self.feature_mean) / np.maximum(self.feature_std, 1e-8)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        feature = sample["feature"].astype(np.float32)
        feature = self._maybe_crop(feature)
        # Descriptor on raw dB energy, before clip/feature normalisation.
        wb_descriptor = (
            compute_wingbeat_descriptor(feature, *self.wb_bins) if self.wb_bins else None
        )
        if self.clip_normalize:
            feature = clip_instance_normalize(feature)
        feature = self._normalize(feature, sample["domain"])
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        if self.training and self.augment is not None:
            feature_tensor = self.augment(feature_tensor)
        item = {
            "file_id": sample["file_id"],
            "feature": feature_tensor,
            "length": feature.shape[0],
            "species_label": sample["species_label"],
            "domain_label": sample["domain_label"],
            "species": sample["species"],
            "domain": sample["domain"],
            "audio_path": sample["audio_path"],
        }
        if wb_descriptor is not None:
            item["wb_descriptor"] = torch.from_numpy(wb_descriptor)
        return item


class LodoFeatureDataset(Dataset):
    """Dataset backed by an in-memory item list rather than a feature pickle.

    Used by ``train_lodo.py`` where items are filtered by domain before loading.
    """

    def __init__(
        self,
        items: List[Dict],
        feature_mean: Optional[np.ndarray],
        feature_std: Optional[np.ndarray],
        max_train_frames: Optional[int],
        training: bool,
        normalize_features: bool,
        max_eval_frames: Optional[int] = None,
        clip_normalize: bool = False,
        augment: Optional[Callable] = None,
        wingbeat_params: Optional[Dict] = None,
        per_domain_norm: bool = False,
    ) -> None:
        self.samples = items
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.max_train_frames = max_train_frames
        self.max_eval_frames = max_eval_frames
        self.training = training
        self.clip_normalize = clip_normalize
        self.augment = augment
        self.wb_bins = wingbeat_band_bins(**wingbeat_params) if wingbeat_params else None
        self.per_domain_norm = per_domain_norm
        self.domain_stats: Optional[Dict[str, tuple]] = None
        if per_domain_norm:
            # Per-domain z-score from this split's own items (transductive).
            self.domain_stats = compute_domain_stats(items)
            self.normalize_features = True
        else:
            self.normalize_features = normalize_features and feature_mean is not None

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_crop(self, feature: np.ndarray) -> np.ndarray:
        if self.training and self.max_train_frames and feature.shape[0] > self.max_train_frames:
            start = random.randint(0, feature.shape[0] - self.max_train_frames)
            return feature[start : start + self.max_train_frames]
        if not self.training and self.max_eval_frames and feature.shape[0] > self.max_eval_frames:
            mid   = feature.shape[0] // 2
            half  = self.max_eval_frames // 2
            start = max(0, mid - half)
            return feature[start : start + self.max_eval_frames]
        return feature

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        feature = sample["feature"].astype(np.float32)          # [T, n_mels]
        feature = self._maybe_crop(feature)

        # Descriptor on raw dB energy, before clip/feature normalisation.
        wb_descriptor = (
            compute_wingbeat_descriptor(feature, *self.wb_bins) if self.wb_bins else None
        )

        if self.clip_normalize:
            feature = clip_instance_normalize(feature)

        if self.per_domain_norm:
            mean, std = self.domain_stats[sample["domain"]]
            feature = (feature - mean) / np.maximum(std, 1e-8)
        elif self.normalize_features:
            feature = (feature - self.feature_mean) / np.maximum(self.feature_std, 1e-8)

        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        if self.training and self.augment is not None:
            feature_tensor = self.augment(feature_tensor)

        item = {
            "file_id":       sample["file_id"],
            "feature":       feature_tensor,
            "length":        feature.shape[0],
            "species_label": sample["species_label"],
            "domain_label":  sample["domain_label"],
            "species":       sample["species"],
            "domain":        sample["domain"],
            "audio_path":    sample["audio_path"],
        }
        if wb_descriptor is not None:
            item["wb_descriptor"] = torch.from_numpy(wb_descriptor)
        return item


def pad_collate_fn(batch: List[Dict]) -> Dict:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item()) if len(lengths) else 0
    feature_dim = int(batch[0]["feature"].shape[1]) if batch else 0
    padded = torch.zeros(len(batch), max_length, feature_dim, dtype=torch.float32)

    for idx, item in enumerate(batch):
        padded[idx, : item["length"], :] = item["feature"]

    collated = {
        "file_id": [item["file_id"] for item in batch],
        "features": padded,
        "lengths": lengths,
        "species_labels": torch.tensor([item["species_label"] for item in batch], dtype=torch.long),
        "domain_labels": torch.tensor([item["domain_label"] for item in batch], dtype=torch.long),
        "species": [item["species"] for item in batch],
        "domain": [item["domain"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
    }
    if batch and "wb_descriptor" in batch[0]:
        collated["wb_descriptor"] = torch.stack([item["wb_descriptor"] for item in batch])
    return collated
