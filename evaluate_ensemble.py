"""Ensemble inference script for submission.

Loads multiple (config, checkpoint) pairs, averages their softmax probabilities
(optionally weighted by LODO BA_unseen), and writes a submission TXT file.

Ensemble members are specified via a JSON file:
    [
      {"config": "configs/lodo_balanced_dann.json",
       "checkpoint": "outputs/.../model/model_best.pth",
       "weight": 0.4,
       "label": "MTRCNN balanced+DANN"},
      ...
    ]

Weights are normalised to sum to 1 automatically. Omit "weight" for equal weighting.

Supports two inference modes:
  --eval-dir  <path>   Unlabelled WAV files (submission mode, default)
  --feature-pkl <path> Pre-extracted feature pickle with labels (evaluation mode,
                       prints per-domain species BA for sanity-checking on test split)

Usage:
    # Submission
    python evaluate_ensemble.py \\
        --members ensemble.json \\
        --eval-dir data/Evaluation_data/ \\
        --out submission_ensemble.txt

    # Sanity check on test split (with labels)
    python evaluate_ensemble.py \\
        --members ensemble.json \\
        --feature-pkl Development_data/feature/test_features.pkl \\
        --out /dev/null
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from framework.acoustic_feature import LogMelSpectrogram, load_waveform
from framework.config import load_config
from framework.dataset import clip_instance_normalize, load_feature_stats
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.utilization import build_model, choose_device, training_stats_path

SPECIES_ID = {name: idx + 1 for idx, name in enumerate(SPECIES_NAMES)}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class EvalAudioDataset(Dataset):
    """Unlabelled WAV files → log-mel on-the-fly (submission mode)."""

    def __init__(self, eval_dir: Path, config: dict) -> None:
        self.paths      = sorted(eval_dir.glob("*.wav"))
        self.config     = config
        self.clip_norm  = config.get("clip_normalize", False)
        self.extractor  = LogMelSpectrogram(
            sample_rate=config["sample_rate"], n_fft=config["n_fft"],
            hop_length=config["hop_length"],  win_length=config["win_length"],
            n_mels=config["n_mels"], fmin=config["fmin"], fmax=config["fmax"],
        )
        stats_path = training_stats_path(config)
        with open(stats_path) as f:
            stats = json.load(f)
        self.feat_mean = np.array(stats["mean"], dtype=np.float32)
        self.feat_std  = np.array(stats["std"],  dtype=np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path  = self.paths[idx]
        wav   = load_waveform(path, self.config["sample_rate"],
                              self.config["normalize_waveform"])
        wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            feat = self.extractor(wav_t)[0].numpy()
        if self.clip_norm:
            feat = clip_instance_normalize(feat)
        if self.config.get("normalize_features", True):
            feat = (feat - self.feat_mean) / np.maximum(self.feat_std, 1e-8)
        return {
            "file_id":        path.stem,
            "feature":        torch.tensor(feat, dtype=torch.float32),
            "length":         feat.shape[0],
            "species_label":  -1,
            "domain_label":   -1,
            "species":        "",
            "domain":         "",
        }


class FeaturePickleDataset(Dataset):
    """Pre-extracted features with labels (evaluation/sanity-check mode)."""

    def __init__(self, pkl_path: Path, config: dict) -> None:
        import pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.items     = data["items"]
        self.clip_norm = config.get("clip_normalize", False)
        stats_path = training_stats_path(config)
        self.feat_mean, self.feat_std = load_feature_stats(stats_path)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        feat = item["feature"].astype(np.float32)
        if self.clip_norm:
            feat = clip_instance_normalize(feat)
        feat = (feat - self.feat_mean) / np.maximum(self.feat_std, 1e-8)
        return {
            "file_id":       item["file_id"],
            "feature":       torch.tensor(feat, dtype=torch.float32),
            "length":        feat.shape[0],
            "species_label": item["species_label"],
            "domain_label":  item["domain_label"],
            "species":       item["species"],
            "domain":        item["domain"],
        }


def collate_fn(batch: List[dict]) -> dict:
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_t   = int(lengths.max())
    n_mels  = batch[0]["feature"].shape[1]
    padded  = torch.zeros(len(batch), max_t, n_mels)
    for i, b in enumerate(batch):
        padded[i, :b["length"]] = b["feature"]
    return {
        "file_ids":       [b["file_id"] for b in batch],
        "features":       padded,
        "lengths":        lengths,
        "species_labels": torch.tensor([b["species_label"] for b in batch], dtype=torch.long),
        "domain_labels":  torch.tensor([b["domain_label"]  for b in batch], dtype=torch.long),
        "species":        [b["species"] for b in batch],
        "domain":         [b["domain"]  for b in batch],
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Run inference, return (file_ids, probs [N,9], species_labels [N], domain_labels [N])."""
    model.eval()
    all_ids, all_probs, all_sp, all_dom = [], [], [], []
    for batch in loader:
        feats   = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        out     = model(feats, lengths)
        probs   = F.softmax(out["species_logits"], dim=-1).cpu().numpy()
        all_ids.extend(batch["file_ids"])
        all_probs.append(probs)
        all_sp.extend(batch["species_labels"].numpy())
        all_dom.extend(batch["domain_labels"].numpy())
    return all_ids, np.concatenate(all_probs), np.array(all_sp), np.array(all_dom)


# ---------------------------------------------------------------------------
# Metrics (when labels available)
# ---------------------------------------------------------------------------

def balanced_accuracy(preds: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        recalls.append((preds[mask] == c).mean())
    return float(np.mean(recalls)) if recalls else 0.0


def print_metrics(
    file_ids: List[str],
    ensemble_probs: np.ndarray,
    species_labels: np.ndarray,
    domain_labels: np.ndarray,
) -> None:
    preds = ensemble_probs.argmax(axis=1)
    overall_ba = balanced_accuracy(preds, species_labels, len(SPECIES_NAMES))
    print(f"\nEnsemble test species BA: {overall_ba:.4f}")

    print("\nPer-domain species BA:")
    for di, dname in enumerate(DOMAIN_NAMES):
        mask = domain_labels == di
        if mask.sum() == 0:
            continue
        ba = balanced_accuracy(preds[mask], species_labels[mask], len(SPECIES_NAMES))
        print(f"  {dname}: {ba:.4f}  (N={mask.sum()})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble inference for submission.")
    parser.add_argument("--members",     required=True,
                        help="JSON file listing ensemble members (config, checkpoint, weight, label)")
    parser.add_argument("--eval-dir",    default="data/Evaluation_data/",
                        help="Directory of unlabelled WAV files (submission mode)")
    parser.add_argument("--feature-pkl", default=None,
                        help="Pre-extracted feature pickle (evaluation mode, has labels)")
    parser.add_argument("--out",         default="submission_ensemble.txt")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # ---- Load ensemble spec --------------------------------------------------
    members: List[dict] = json.loads(Path(args.members).read_text())
    weights = np.array([m.get("weight", 1.0) for m in members], dtype=np.float32)
    weights /= weights.sum()
    print(f"Ensemble: {len(members)} members")
    for m, w in zip(members, weights):
        print(f"  [{w:.3f}] {m.get('label', m['checkpoint'])}")

    # ---- Build dataset -------------------------------------------------------
    # Use config from first member for dataset construction (feature extraction
    # settings must be identical across all members — enforced by config_signature).
    first_cfg = load_config(members[0]["config"])
    device    = choose_device(first_cfg["device"])

    if args.feature_pkl:
        dataset = FeaturePickleDataset(Path(args.feature_pkl), first_cfg)
        mode    = "eval_with_labels"
        print(f"\nMode: evaluation (feature pickle, N={len(dataset)})")
    else:
        dataset = EvalAudioDataset(Path(args.eval_dir), first_cfg)
        mode    = "submission"
        print(f"\nMode: submission (WAV dir, N={len(dataset)})")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # ---- Run each member -----------------------------------------------------
    ensemble_probs: Optional[np.ndarray] = None
    file_ids = species_labels = domain_labels = None

    for member, w in zip(members, weights):
        cfg   = load_config(member["config"])
        label = member.get("label", Path(member["checkpoint"]).parent.parent.name)
        print(f"\n--- {label} (weight={w:.3f}) ---")

        model = build_model(cfg, device)
        ckpt  = torch.load(member["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        ids, probs, sp_labels, dom_labels = predict_probs(model, loader, device)

        # Per-member accuracy when labels available
        if mode == "eval_with_labels" and sp_labels[0] >= 0:
            preds = probs.argmax(axis=1)
            ba = balanced_accuracy(preds, sp_labels, len(SPECIES_NAMES))
            print(f"  Individual BA: {ba:.4f}")

        if ensemble_probs is None:
            ensemble_probs = w * probs
            file_ids, species_labels, domain_labels = ids, sp_labels, dom_labels
        else:
            ensemble_probs += w * probs

        del model
        torch.cuda.empty_cache()

    # ---- Ensemble metrics ----------------------------------------------------
    if mode == "eval_with_labels" and species_labels[0] >= 0:
        print_metrics(file_ids, ensemble_probs, species_labels, domain_labels)

    # ---- Write submission ----------------------------------------------------
    out_path = Path(args.out)
    if str(out_path) != "/dev/null":
        preds = ensemble_probs.argmax(axis=1)
        with open(out_path, "w") as f:
            for fid, pred_idx in zip(file_ids, preds):
                f.write(f"{fid} {pred_idx + 1}\n")   # 1-based species ID

        from collections import Counter
        counts = Counter(preds + 1)
        print(f"\nSubmission written: {out_path} ({len(file_ids)} clips)")
        print("Predicted species distribution:")
        for sid in sorted(counts):
            print(f"  Species {sid} ({SPECIES_NAMES[sid-1]}): {counts[sid]}")


if __name__ == "__main__":
    main()
