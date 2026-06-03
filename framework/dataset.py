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
        clip_normalize: bool = False,
        expected_feature_signature: Optional[str] = None,
        expected_stats_signature: Optional[str] = None,
        augment: Optional[Callable] = None,
    ) -> None:
        payload = load_feature_payload(feature_pickle_path)
        validate_feature_payload(payload, expected_feature_signature)
        self.samples = payload["items"]
        self.training = training
        self.max_train_frames = max_train_frames
        self.normalize_features = normalize_features and feature_stats_path is not None
        self.clip_normalize = clip_normalize
        self.feature_mean = None
        self.feature_std = None
        self.augment = augment
        if self.normalize_features:
            validate_feature_stats_payload(feature_stats_path, expected_stats_signature)
            self.feature_mean, self.feature_std = load_feature_stats(feature_stats_path)

    def __len__(self) -> int:
        return len(self.samples)

    def _maybe_crop(self, feature: np.ndarray) -> np.ndarray:
        if not self.training or not self.max_train_frames or feature.shape[0] <= self.max_train_frames:
            return feature
        start = random.randint(0, feature.shape[0] - self.max_train_frames)
        return feature[start : start + self.max_train_frames]

    def _normalize(self, feature: np.ndarray) -> np.ndarray:
        if not self.normalize_features:
            return feature
        return (feature - self.feature_mean) / np.maximum(self.feature_std, 1e-8)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        feature = sample["feature"].astype(np.float32)
        feature = self._maybe_crop(feature)
        if self.clip_normalize:
            feature = clip_instance_normalize(feature)
        feature = self._normalize(feature)
        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        if self.training and self.augment is not None:
            feature_tensor = self.augment(feature_tensor)
        return {
            "file_id": sample["file_id"],
            "feature": feature_tensor,
            "length": feature.shape[0],
            "species_label": sample["species_label"],
            "domain_label": sample["domain_label"],
            "species": sample["species"],
            "domain": sample["domain"],
            "audio_path": sample["audio_path"],
        }


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
        clip_normalize: bool = False,
        augment: Optional[Callable] = None,
    ) -> None:
        self.samples = items
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.max_train_frames = max_train_frames
        self.training = training
        self.normalize_features = normalize_features and feature_mean is not None
        self.clip_normalize = clip_normalize
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        feature = sample["feature"].astype(np.float32)          # [T, n_mels]

        if self.training and self.max_train_frames and feature.shape[0] > self.max_train_frames:
            start = random.randint(0, feature.shape[0] - self.max_train_frames)
            feature = feature[start : start + self.max_train_frames]

        if self.clip_normalize:
            feature = clip_instance_normalize(feature)

        if self.normalize_features:
            feature = (feature - self.feature_mean) / np.maximum(self.feature_std, 1e-8)

        feature_tensor = torch.tensor(feature, dtype=torch.float32)
        if self.training and self.augment is not None:
            feature_tensor = self.augment(feature_tensor)

        return {
            "file_id":       sample["file_id"],
            "feature":       feature_tensor,
            "length":        feature.shape[0],
            "species_label": sample["species_label"],
            "domain_label":  sample["domain_label"],
            "species":       sample["species"],
            "domain":        sample["domain"],
            "audio_path":    sample["audio_path"],
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
