#!/usr/bin/env python3
"""Extract frozen Perch v2 embeddings from raw audio.

Perch v2 (bird-vocalization-classifier/4 on TF Hub) produces one 1280-dim
embedding per 5-second 32 kHz audio window.  For each mosquito clip this
script:

  1. Loads audio at the config sample_rate (8 kHz) via librosa.
  2. Resamples to 32 kHz (Perch's expected input rate).
  3. Segments into non-overlapping 5-second windows; pads the last window
     with zeros if the clip is shorter than 5 s.
  4. Runs the frozen Perch encoder — one 1280-d vector per window.
  5. Saves [n_windows, 1280] arrays in the same pickle format used by
     the baseline log-mel pipeline, so train_lodo.py works unchanged.

Dependencies (not in base requirements.txt)
-------------------------------------------
    pip install tensorflow-cpu tensorflow-hub

    # GPU (optional, extraction can run on CPU):
    pip install tensorflow tensorflow-hub

Usage
-----
    python extract_perch_features.py --config configs/lodo_perch.json
    python extract_perch_features.py --config configs/lodo_perch.json --overwrite

Author: Sulagna Saha
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np

from framework.config import config_signature, load_config
from framework.metadata import DOMAIN_TO_INDEX, SPECIES_TO_INDEX, load_id_list, parse_file_id

# ---------------------------------------------------------------------------
# Perch constants
# ---------------------------------------------------------------------------

PERCH_SAMPLE_RATE    = 32_000                    # Hz
PERCH_EMBED_DIM      = 1_280
PERCH_WINDOW_SAMPLES = 5 * PERCH_SAMPLE_RATE     # 160 000 samples per 5-second window
DEFAULT_MODEL_URL    = "https://tfhub.dev/google/bird-vocalization-classifier/4"


# ---------------------------------------------------------------------------
# TF dependency guard
# ---------------------------------------------------------------------------

def _require_tf():
    try:
        import tensorflow as tf          # noqa: F401
        import tensorflow_hub as hub     # noqa: F401
        return tf, hub
    except ImportError:
        raise SystemExit(
            "\nTensorFlow and TensorFlow-Hub are required for Perch feature extraction.\n"
            "  pip install tensorflow-cpu tensorflow-hub\n"
            "(Use 'tensorflow' instead of 'tensorflow-cpu' if you want GPU support.)"
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_perch_model(model_url: str):
    """Download and load Perch from TF Hub (cached after first download).

    Returns the serving_default concrete function so callers use a unified
    interface regardless of TF Hub version.

    Signature: fn(inputs=[batch, 160000]) → {"output_0": logits [batch, 10932],
                                               "output_1": embedding [batch, 1280]}
    """
    tf, hub = _require_tf()
    print(f"Loading Perch model: {model_url}")
    model = hub.load(model_url)
    infer_fn = model.signatures["serving_default"]
    print("Perch model ready.")
    return infer_fn, tf


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings(
    waveform_32k: np.ndarray,
    model,
    tf,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract Perch embeddings for a single variable-length 32 kHz clip.

    Segments the clip into non-overlapping 5-second windows (zero-pads the
    last window when shorter), runs a batched forward pass, and returns the
    stacked per-window embeddings.

    Args:
        waveform_32k: 1-D float32 waveform at 32 kHz.
        model:        loaded TF Hub Perch model.
        tf:           tensorflow module reference.
        batch_size:   max windows per forward pass (memory control).

    Returns:
        embeddings: float32 array of shape [n_windows, 1280].
    """
    n_samples = len(waveform_32k)
    n_windows = max(1, (n_samples + PERCH_WINDOW_SAMPLES - 1) // PERCH_WINDOW_SAMPLES)
    padded_len = n_windows * PERCH_WINDOW_SAMPLES
    if n_samples < padded_len:
        waveform_32k = np.pad(waveform_32k, (0, padded_len - n_samples))

    windows = waveform_32k.reshape(n_windows, PERCH_WINDOW_SAMPLES)  # [n_w, 160000]

    parts: List[np.ndarray] = []
    for start in range(0, n_windows, batch_size):
        chunk = tf.constant(windows[start : start + batch_size], dtype=tf.float32)
        outputs = model(inputs=chunk)
        parts.append(outputs["output_1"].numpy())   # output_1 = embedding [batch, 1280]

    return np.concatenate(parts, axis=0).astype(np.float32)  # [n_windows, 1280]


# ---------------------------------------------------------------------------
# Signature helpers
# ---------------------------------------------------------------------------

def perch_signature_payload(config: Dict, split_name: str) -> Dict:
    """Config payload for the Perch feature signature (log-mel keys excluded)."""
    ids_key = {
        "training":   "train_ids_path",
        "validation": "val_ids_path",
        "test":       "test_ids_path",
    }[split_name]
    from framework.config import file_sha256
    return {
        "dataset_root":    config["dataset_root"],
        "sample_rate":     config["sample_rate"],
        "normalize_waveform": config["normalize_waveform"],
        "perch_model_url": config.get("perch_model_url", DEFAULT_MODEL_URL),
        "split":           split_name,
        "ids_path":        config[ids_key],
        "ids_sha256":      file_sha256(config[ids_key]),
    }


# ---------------------------------------------------------------------------
# Per-split extraction
# ---------------------------------------------------------------------------

def extract_split(
    config: Dict,
    split_name: str,
    model,
    tf,
    overwrite: bool,
) -> Path:
    feature_root = Path(config["feature_root"])
    feature_root.mkdir(parents=True, exist_ok=True)
    output_path = feature_root / f"{split_name.lower()}_features.pkl"

    sig = config_signature(perch_signature_payload(config, split_name))

    # Skip if already extracted with the same config.
    if output_path.exists() and not overwrite:
        with open(output_path, "rb") as fh:
            stored = pickle.load(fh)
        if stored.get("config_signature") == sig:
            print(f"[{split_name}] cached — skipping (pass --overwrite to force)")
            return output_path
        print(f"[{split_name}] config changed — re-extracting")

    ids_key  = {"training": "train_ids_path", "validation": "val_ids_path", "test": "test_ids_path"}[split_name]
    file_ids = load_id_list(config[ids_key])
    records: List[Dict] = []

    for idx, file_id in enumerate(file_ids, 1):
        species, domain = parse_file_id(file_id)
        audio_path = Path(config["dataset_root"]) / f"{file_id}.wav"

        waveform, _ = librosa.load(str(audio_path), sr=config["sample_rate"], mono=True)
        if config.get("normalize_waveform", True) and waveform.size:
            peak = np.abs(waveform).max()
            if peak > 0:
                waveform = waveform / peak

        # Resample 8 kHz → 32 kHz (or whatever source sr → PERCH_SAMPLE_RATE).
        waveform_32k = librosa.resample(
            waveform, orig_sr=config["sample_rate"], target_sr=PERCH_SAMPLE_RATE
        ).astype(np.float32)

        embeddings = extract_embeddings(waveform_32k, model, tf)  # [n_windows, 1280]

        print(
            f"[{split_name}] {idx}/{len(file_ids)} | id={file_id} | "
            f"n_windows={embeddings.shape[0]}"
        )
        records.append({
            "file_id":       file_id,
            "feature":       embeddings,          # [n_windows, 1280]  float32
            "num_frames":    embeddings.shape[0],
            "feature_dim":   embeddings.shape[1],
            "species":       species,
            "species_label": SPECIES_TO_INDEX[species],
            "domain":        domain,
            "domain_label":  DOMAIN_TO_INDEX[domain],
            "audio_path":    str(audio_path),
        })

    payload = {
        "split":            split_name,
        "num_items":        len(records),
        "config_signature": sig,
        "items":            records,
    }
    with open(output_path, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[{split_name}] saved {len(records)} items → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Perch v2 embeddings.")
    parser.add_argument("--config",    required=True, help="Path to experiment JSON config.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if cached.")
    args = parser.parse_args()

    config = load_config(args.config)
    if "perch_model_url" not in config:
        config["perch_model_url"] = DEFAULT_MODEL_URL

    model, tf = load_perch_model(config["perch_model_url"])

    for split in ("training", "validation", "test"):
        extract_split(config, split, model, tf, args.overwrite)

    print("Done.")


if __name__ == "__main__":
    main()
