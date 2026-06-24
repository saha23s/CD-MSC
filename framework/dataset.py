"""Dataset helpers for reading precomputed feature files.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import median_filter
from torch.utils.data import Dataset


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


class MosquitoFeatureDataset(Dataset):
    def __init__(
        self,
        feature_pickle_path: Union[str, Path],
        feature_stats_path: Optional[Union[str, Path]] = None,
        max_train_frames: Optional[int] = None,
        training: bool = False,
        normalize_features: bool = True,
        expected_feature_signature: Optional[str] = None,
        expected_stats_signature: Optional[str] = None,
        spec_augment: bool = False,
        spec_augment_time_mask: int = 40,
        spec_augment_freq_mask: int = 10,
        cmn: bool = False,
        d5_noise_std: float = 0.0,
        use_delta: bool = False,
        freq_shift_bins: int = 0,
        use_approx_hpss: bool = False,
        hist_match: bool = False,
        domain_stats: Optional[Dict] = None,
        use_fda: bool = False,
        fda_beta: float = 0.05,
        fda_prob: float = 0.5,
        dirus_freq_shift_bins: int = 0,
        dirus_temporal_dropout: float = 0.0,
    ) -> None:
        payload = load_feature_payload(feature_pickle_path)
        validate_feature_payload(payload, expected_feature_signature)
        self.samples = payload["items"]
        self.training = training
        self.max_train_frames = max_train_frames
        self.normalize_features = normalize_features and feature_stats_path is not None
        self.spec_augment = spec_augment and training
        self.spec_augment_time_mask = spec_augment_time_mask
        self.spec_augment_freq_mask = spec_augment_freq_mask
        self.cmn = cmn
        self.d5_noise_std = d5_noise_std
        self.use_delta = use_delta
        self.freq_shift_bins = freq_shift_bins
        self.use_approx_hpss = use_approx_hpss
        self.hist_match = hist_match and training
        self.domain_stats = domain_stats
        self.use_fda = use_fda and training
        self.fda_beta = fda_beta
        self.fda_prob = fda_prob
        self.dirus_freq_shift_bins = dirus_freq_shift_bins
        self.dirus_temporal_dropout = dirus_temporal_dropout
        # Pre-index field (D1–D4) samples for fast random access during FDA augmentation.
        # domain_label==4 is D5 (lab); 0–3 are field domains.
        self.field_indices: List[int] = []
        if self.use_fda:
            self.field_indices = [i for i, s in enumerate(self.samples) if s["domain_label"] != 4]
        self.feature_mean = None
        self.feature_std = None
        if self.normalize_features:
            validate_feature_stats_payload(feature_stats_path, expected_stats_signature)
            self.feature_mean, self.feature_std = load_feature_stats(feature_stats_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _spec_augment(self, feature: np.ndarray) -> np.ndarray:
        T, F = feature.shape
        out = feature.copy()
        t = random.randint(0, self.spec_augment_time_mask)
        if t > 0 and T > t:
            t0 = random.randint(0, T - t)
            out[t0:t0 + t, :] = 0.0
        f = random.randint(0, self.spec_augment_freq_mask)
        if f > 0 and F > f:
            f0 = random.randint(0, F - f)
            out[:, f0:f0 + f] = 0.0
        return out

    def _maybe_crop(self, feature: np.ndarray) -> np.ndarray:
        if not self.training or not self.max_train_frames or feature.shape[0] <= self.max_train_frames:
            return feature
        start = random.randint(0, feature.shape[0] - self.max_train_frames)
        return feature[start : start + self.max_train_frames]

    def _normalize(self, feature: np.ndarray) -> np.ndarray:
        if not self.normalize_features:
            return feature
        return (feature - self.feature_mean) / np.maximum(self.feature_std, 1e-8)

    def _cmn(self, feature: np.ndarray) -> np.ndarray:
        # Subtract per-clip time-axis mean from each mel bin — removes channel/mic offset
        return feature - feature.mean(axis=0, keepdims=True)

    def _d5_noise(self, feature: np.ndarray) -> np.ndarray:
        # Add Gaussian noise to D5 (lab) clips during training to simulate field SNR
        return feature + np.random.normal(0.0, self.d5_noise_std, feature.shape).astype(np.float32)

    def _dirus_freq_shift(self, feature: np.ndarray) -> np.ndarray:
        # Shift An. dirus D1 training clips downward in frequency to simulate D4 conditions.
        # D1 median F0 ~793 Hz; D4 median F0 ~307 Hz — a shift of ~13 mel bins downward.
        # Negative roll moves content toward lower-frequency bins.
        return np.roll(feature, self.dirus_freq_shift_bins, axis=1)

    def _dirus_temporal_dropout(self, feature: np.ndarray) -> np.ndarray:
        # Randomly silence a fraction of frames in An. dirus D1 clips to simulate intermittent
        # flight bursts observed in D4 field recordings (voiced fraction ~0.34 vs ~1.0 in D1).
        T = feature.shape[0]
        n_drop = int(T * self.dirus_temporal_dropout)
        if n_drop > 0:
            drop_frames = np.random.choice(T, size=n_drop, replace=False)
            out = feature.copy()
            out[drop_frames, :] = 0.0
            return out
        return feature

    def _freq_shift(self, feature: np.ndarray) -> np.ndarray:
        shift = random.randint(-self.freq_shift_bins, self.freq_shift_bins)
        return np.roll(feature, shift, axis=1)

    def _approx_hpss(self, feature: np.ndarray) -> np.ndarray:
        # Wiener masking on log-mel: harmonic = time-persistent, percussive = freq-spread
        H = median_filter(feature, size=(17, 1))  # large time kernel → harmonic estimate
        P = median_filter(feature, size=(1, 9))   # large freq kernel → percussive estimate
        H2, P2 = H ** 2, P ** 2
        return feature * H2 / (H2 + P2 + 1e-6)

    def _fda_augment(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        # Fourier Domain Adaptation: swap the low-frequency 2D amplitude components of a
        # D5 (lab) clip with those of a field (D1–D4) clip from any species.
        # The source species label is preserved — only the coarse acoustic envelope changes.
        # beta controls the fraction of each frequency axis that is swapped; small values
        # (~0.05) target only room acoustics / spectral coloration, leaving the high-frequency
        # wingbeat oscillations intact.
        T, F = source.shape

        # Align target length to source by random crop or edge-pad.
        Tt = target.shape[0]
        if Tt >= T:
            start = random.randint(0, Tt - T)
            target = target[start : start + T]
        else:
            target = np.pad(target, ((0, T - Tt), (0, 0)), mode="edge")

        fft_s = np.fft.fft2(source)
        fft_t = np.fft.fft2(target)

        amp_s = np.fft.fftshift(np.abs(fft_s))
        amp_t = np.fft.fftshift(np.abs(fft_t))
        phase_s = np.angle(fft_s)

        # Swap a central (low-frequency) window in the shifted amplitude spectrum.
        h_cut = max(1, int(np.ceil(T * self.fda_beta)))
        w_cut = max(1, int(np.ceil(F * self.fda_beta)))
        h_ctr, w_ctr = T // 2, F // 2
        amp_new = amp_s.copy()
        amp_new[h_ctr - h_cut : h_ctr + h_cut, w_ctr - w_cut : w_ctr + w_cut] = \
            amp_t[h_ctr - h_cut : h_ctr + h_cut, w_ctr - w_cut : w_ctr + w_cut]
        amp_new = np.fft.ifftshift(amp_new)

        fft_new = amp_new * np.exp(1j * phase_s)
        return np.real(np.fft.ifft2(fft_new)).astype(np.float32)

    def _domain_hist_match(self, feature: np.ndarray) -> np.ndarray:
        # Shift D5 clip distribution to a randomly chosen field domain (D1–D4)
        target = random.choice(["D1", "D2", "D3", "D4"])
        src = self.domain_stats["D5"]
        tgt = self.domain_stats[target]
        mean_s = np.array(src["mean"], dtype=np.float32)
        std_s  = np.array(src["std"],  dtype=np.float32)
        mean_t = np.array(tgt["mean"], dtype=np.float32)
        std_t  = np.array(tgt["std"],  dtype=np.float32)
        return (feature - mean_s) / (std_s + 1e-8) * std_t + mean_t

    def _compute_delta(self, feature: np.ndarray) -> np.ndarray:
        # Compute first-order time-axis delta and concatenate with mel along frequency axis.
        # Delta is computed on already-normalised mel so global mean cancels in the difference.
        # Result: [T, 2*F] where first F cols are mel, next F cols are temporal differences.
        delta = np.diff(feature, axis=0, prepend=feature[:1]).astype(np.float32)
        return np.concatenate([feature, delta], axis=1)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        feature = sample["feature"].astype(np.float32)
        feature = self._maybe_crop(feature)
        if self.use_fda and sample["domain_label"] == 4 and self.field_indices and random.random() < self.fda_prob:
            target_idx = random.choice(self.field_indices)
            target_feature = self.samples[target_idx]["feature"].astype(np.float32)
            feature = self._fda_augment(feature, target_feature)
        if self.use_approx_hpss:
            feature = self._approx_hpss(feature)
        if self.hist_match and sample["domain_label"] == 4 and self.domain_stats is not None:
            feature = self._domain_hist_match(feature)
        feature = self._normalize(feature)
        if self.cmn:
            feature = self._cmn(feature)
        if self.training and self.d5_noise_std > 0.0 and sample["domain_label"] == 4:
            feature = self._d5_noise(feature)
        if self.training and self.freq_shift_bins > 0 and sample["domain_label"] == 4:
            feature = self._freq_shift(feature)
        # An. dirus D1 augmentation: simulate D4 recording conditions.
        # species_label==5 is An. dirus (0-indexed); domain_label==0 is D1.
        _is_dirus_d1 = sample["species_label"] == 5 and sample["domain_label"] == 0
        if self.training and self.dirus_freq_shift_bins != 0 and _is_dirus_d1:
            feature = self._dirus_freq_shift(feature)
        if self.training and self.dirus_temporal_dropout > 0.0 and _is_dirus_d1:
            feature = self._dirus_temporal_dropout(feature)
        if self.use_delta:
            feature = self._compute_delta(feature)
        if self.spec_augment:
            feature = self._spec_augment(feature)
        return {
            "file_id": sample["file_id"],
            "feature": torch.tensor(feature, dtype=torch.float32),
            "length": feature.shape[0],
            "species_label": sample["species_label"],
            "domain_label": sample["domain_label"],
            "species": sample["species"],
            "domain": sample["domain"],
            "audio_path": sample["audio_path"],
        }


def pad_collate_fn(batch: List[Dict]) -> Dict:
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_length = int(lengths.max().item()) if len(lengths) else 0
    feature_dim = int(batch[0]["feature"].shape[1]) if batch else 0
    padded = torch.zeros(len(batch), max_length, feature_dim, dtype=torch.float32)

    for idx, item in enumerate(batch):
        padded[idx, : item["length"], :] = item["feature"]

    return {
        "file_id": [item["file_id"] for item in batch],
        "features": padded,
        "lengths": lengths,
        "species_labels": torch.tensor([item["species_label"] for item in batch], dtype=torch.long),
        "domain_labels": torch.tensor([item["domain_label"] for item in batch], dtype=torch.long),
        "species": [item["species"] for item in batch],
        "domain": [item["domain"] for item in batch],
        "audio_path": [item["audio_path"] for item in batch],
    }
