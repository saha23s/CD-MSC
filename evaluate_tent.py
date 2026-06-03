"""Test-time entropy minimisation (TENT) on the evaluation set + submission file.

Loads a trained checkpoint, adapts BN/LN affine parameters to the evaluation
clips by minimising prediction entropy (no labels used), then writes a TXT
submission file.

Usage:
    python evaluate_tent.py \\
        --checkpoint outputs/<run>/model/model_best.pth \\
        --eval-dir   data/Evaluation_data/ \\
        --out        submission_tent.txt \\
        [--config    configs/default_experiment.json] \\
        [--tent-lr   1e-3] \\
        [--tent-steps 3] \\
        [--batch-size 64]

The submission format is one row per clip:
    <file_id> <predicted_species_id>

where file_id is the wav stem (e.g. CDMSC2026_EVAL_000001) and
predicted_species_id is the 1-based integer matching the challenge schema.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from framework.acoustic_feature import LogMelSpectrogram, load_waveform
from framework.config import load_config
from framework.metadata import SPECIES_NAMES
from framework.tent import configure_tent, tent_step
from framework.utilization import build_model, choose_device


# Challenge uses 1-based species IDs matching the order in SPECIES_NAMES
SPECIES_ID = {name: idx + 1 for idx, name in enumerate(SPECIES_NAMES)}


# ---------------------------------------------------------------------------
# Eval dataset (raw audio → log-mel on-the-fly)
# ---------------------------------------------------------------------------

class EvalAudioDataset(Dataset):
    """Loads unlabelled eval WAV files and extracts log-mel on the fly."""

    def __init__(self, eval_dir: Path, config: dict, clip_normalize: bool = False) -> None:
        self.paths = sorted(eval_dir.glob("*.wav"))
        self.config = config
        self.clip_normalize = clip_normalize
        self.extractor = LogMelSpectrogram(
            sample_rate=config["sample_rate"],
            n_fft=config["n_fft"],
            hop_length=config["hop_length"],
            win_length=config["win_length"],
            n_mels=config["n_mels"],
            fmin=config["fmin"],
            fmax=config["fmax"],
        )
        # Load training feature stats for global normalisation
        import json as _json
        from framework.utilization import training_stats_path
        stats_path = training_stats_path(config)
        with open(stats_path) as f:
            stats = _json.load(f)
        self.feat_mean = np.array(stats["mean"], dtype=np.float32)
        self.feat_std  = np.array(stats["std"],  dtype=np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path   = self.paths[idx]
        wav    = load_waveform(path, self.config["sample_rate"],
                               self.config["normalize_waveform"])
        wav_t  = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            feat = self.extractor(wav_t)[0].numpy()         # [T, F]

        if self.clip_normalize:
            from framework.dataset import clip_instance_normalize
            feat = clip_instance_normalize(feat)

        if self.config.get("normalize_features", True):
            feat = (feat - self.feat_mean) / np.maximum(self.feat_std, 1e-8)

        return {
            "file_id": path.stem,
            "feature": torch.tensor(feat, dtype=torch.float32),
            "length":  feat.shape[0],
        }


def collate_fn(batch):
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_t   = int(lengths.max())
    n_mels  = batch[0]["feature"].shape[1]
    padded  = torch.zeros(len(batch), max_t, n_mels)
    for i, b in enumerate(batch):
        padded[i, :b["length"]] = b["feature"]
    return {
        "file_ids": [b["file_id"] for b in batch],
        "features": padded,
        "lengths":  lengths,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--eval-dir",    default="data/Evaluation_data/")
    parser.add_argument("--out",         default="submission_tent.txt")
    parser.add_argument("--config",      default="configs/default_experiment.json")
    parser.add_argument("--tent-lr",     type=float, default=1e-3)
    parser.add_argument("--tent-steps",  type=int,   default=3,
                        help="Gradient steps per batch during TENT adaptation")
    parser.add_argument("--tent-epochs", type=int,   default=1,
                        help="Full passes over eval set during TENT adaptation")
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--no-tent",     action="store_true",
                        help="Skip TENT — just run normal inference (baseline comparison)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = choose_device(config["device"])

    eval_dir = Path(args.eval_dir)
    print(f"Eval clips: {len(list(eval_dir.glob('*.wav')))}")

    dataset    = EvalAudioDataset(eval_dir, config,
                                  clip_normalize=config.get("clip_normalize", False))
    loader     = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn)

    model = build_model(config, device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ---- TENT adaptation pass -----------------------------------------------
    if not args.no_tent:
        optimizer = configure_tent(model, lr=args.tent_lr)
        print(f"Running TENT: {args.tent_epochs} epoch(s), "
              f"{args.tent_steps} steps/batch, lr={args.tent_lr}")
        for epoch in range(args.tent_epochs):
            total_h = 0.0
            for batch in loader:
                features = batch["features"].to(device)
                lengths  = batch["lengths"].to(device)
                h = tent_step(model, features, lengths, optimizer, args.tent_steps)
                total_h += h
            print(f"  Epoch {epoch+1}: mean entropy = {total_h/len(loader):.4f}")

    # ---- Inference pass ------------------------------------------------------
    model.eval()
    # Re-enable running stats for inference (don't use noisy single-batch stats)
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            m.track_running_stats = True
            m.eval()

    predictions = {}
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            lengths  = batch["lengths"].to(device)
            out      = model(features, lengths)
            species_indices = out["species_logits"].argmax(dim=-1).cpu().numpy()
            for fid, idx in zip(batch["file_ids"], species_indices):
                predictions[fid] = idx + 1   # 1-based species ID

    # ---- Write submission file -----------------------------------------------
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        for fid in sorted(predictions):
            f.write(f"{fid} {predictions[fid]}\n")

    print(f"\nSubmission written: {out_path}  ({len(predictions)} clips)")
    # Summary
    from collections import Counter
    species_counts = Counter(predictions.values())
    print("Predicted species distribution (1-based ID → count):")
    for sid, cnt in sorted(species_counts.items()):
        print(f"  Species {sid} ({SPECIES_NAMES[sid-1]}): {cnt}")


if __name__ == "__main__":
    main()
