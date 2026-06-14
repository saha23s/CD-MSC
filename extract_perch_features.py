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

def _segment_windows(waveform_32k: np.ndarray) -> np.ndarray:
    """Segment a 32 kHz clip into non-overlapping 5-second windows.

    Zero-pads the final window when the clip is shorter than a whole number
    of 5-second windows. Mosquito clips are typically < 5 s, so this usually
    returns a single padded window.

    Args:
        waveform_32k: 1-D float32 waveform at 32 kHz.

    Returns:
        windows: float32 array of shape [n_windows, 160000].
    """
    n_samples = len(waveform_32k)
    n_windows = max(1, (n_samples + PERCH_WINDOW_SAMPLES - 1) // PERCH_WINDOW_SAMPLES)
    padded_len = n_windows * PERCH_WINDOW_SAMPLES
    if n_samples < padded_len:
        waveform_32k = np.pad(waveform_32k, (0, padded_len - n_samples))
    return waveform_32k.reshape(n_windows, PERCH_WINDOW_SAMPLES)  # [n_windows, 160000]


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
    limit: Optional[int] = None,
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
    if limit is not None:
        file_ids = file_ids[:limit]
    batch_size = int(config.get("perch_batch_size", 64))
    checkpoint_every = int(config.get("perch_checkpoint_every", 5000))
    partial_path = feature_root / f"{split_name.lower()}_features.partial.pkl"

    def save_pickle(path: Path, payload: Dict) -> None:
        """Atomic pickle write — dump to a temp file then rename, so a crash
        mid-write can never leave a truncated/corrupt file behind."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    # ``records`` holds clips whose features are fully assembled and durably
    # checkpointed. Resume from a matching partial checkpoint if one exists so
    # a timed-out/killed run picks up where it left off instead of restarting.
    records: List[Dict] = []
    done_ids: set = set()
    if partial_path.exists() and not overwrite:
        try:
            with open(partial_path, "rb") as fh:
                part = pickle.load(fh)
            if part.get("config_signature") == sig:
                records = part["items"]
                done_ids = {r["file_id"] for r in records}
                print(f"[{split_name}] resuming from checkpoint: "
                      f"{len(records)}/{len(file_ids)} clips already done")
            else:
                print(f"[{split_name}] partial checkpoint config mismatch — restarting")
        except Exception as exc:  # corrupt/unreadable partial → start fresh
            print(f"[{split_name}] partial checkpoint unreadable ({exc}) — restarting")

    # In-progress clips (windowed but not yet checkpointed). Kept separate from
    # ``records`` so checkpoint boundaries reset their indexing cleanly.
    pend_records: List[Dict] = []
    pend_embeds:  List[List[np.ndarray]] = []   # per-clip list of [1280] rows

    # Streaming window buffer for cross-clip batching: most clips are a single
    # 5-s window, so batching at clip granularity (the old path) ran Perch at
    # batch size 1. Here we pool windows across clips and flush full batches.
    buf_windows: List[np.ndarray] = []          # each [160000]
    buf_owner:   List[int] = []                  # in-progress clip index per window

    def run_batch(n: int) -> None:
        """Run Perch on the first ``n`` buffered windows; scatter to owners."""
        batch = np.stack(buf_windows[:n], axis=0)                       # [n, 160000]
        outputs = model(inputs=tf.constant(batch, dtype=tf.float32))
        emb = outputs["output_1"].numpy().astype(np.float32)           # [n, 1280]
        for i in range(n):
            pend_embeds[buf_owner[i]].append(emb[i])
        del buf_windows[:n]
        del buf_owner[:n]

    def flush_pending() -> None:
        """Assemble in-progress clips into ``records``. Buffer must be drained
        first so every pending clip has all its window embeddings."""
        for rec, rows in zip(pend_records, pend_embeds):
            feature = np.stack(rows, axis=0)      # [n_windows, 1280]
            rec["feature"]     = feature
            rec["feature_dim"] = feature.shape[1]
            records.append(rec)
        pend_records.clear()
        pend_embeds.clear()

    for idx, file_id in enumerate(file_ids, 1):
        if file_id in done_ids:
            continue                              # already checkpointed on a prior run
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

        windows = _segment_windows(waveform_32k)   # [n_windows, 160000]
        clip_idx = len(pend_records)
        pend_embeds.append([])
        pend_records.append({
            "file_id":       file_id,
            "feature":       None,                # filled after batched inference
            "num_frames":    windows.shape[0],
            "feature_dim":   PERCH_EMBED_DIM,
            "species":       species,
            "species_label": SPECIES_TO_INDEX[species],
            "domain":        domain,
            "domain_label":  DOMAIN_TO_INDEX[domain],
            "audio_path":    str(audio_path),
        })
        for w in windows:
            buf_windows.append(w)
            buf_owner.append(clip_idx)

        # Flush whole batches as they accumulate.
        while len(buf_windows) >= batch_size:
            run_batch(batch_size)

        # Periodic crash-safe checkpoint: drain the buffer so all pending clips
        # are complete, assemble them, and durably write the partial pickle.
        if idx % checkpoint_every == 0:
            while buf_windows:
                run_batch(min(batch_size, len(buf_windows)))
            flush_pending()
            save_pickle(partial_path, {
                "split": split_name, "num_items": len(records),
                "config_signature": sig, "items": records,
            })
            print(f"[{split_name}] checkpoint: {len(records)}/{len(file_ids)} clips saved")
        elif idx % 1000 == 0 or idx == len(file_ids):
            print(f"[{split_name}] {idx}/{len(file_ids)} clips windowed")

    # Drain any remaining partial batch and assemble the last clips.
    while buf_windows:
        run_batch(min(batch_size, len(buf_windows)))
    flush_pending()

    save_pickle(output_path, {
        "split":            split_name,
        "num_items":        len(records),
        "config_signature": sig,
        "items":            records,
    })
    if partial_path.exists():
        partial_path.unlink()                     # final file is durable; drop checkpoint
    print(f"[{split_name}] saved {len(records)} items → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen Perch v2 embeddings.")
    parser.add_argument("--config",    required=True, help="Path to experiment JSON config.")
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if cached.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N clips per split (smoke testing).")
    args = parser.parse_args()

    config = load_config(args.config)
    if "perch_model_url" not in config:
        config["perch_model_url"] = DEFAULT_MODEL_URL

    model, tf = load_perch_model(config["perch_model_url"])

    for split in ("training", "validation", "test"):
        extract_split(config, split, model, tf, args.overwrite, limit=args.limit)

    print("Done.")


if __name__ == "__main__":
    main()
