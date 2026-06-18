#!/usr/bin/env python3
"""Evaluate a LODO checkpoint on the official test split using its own per-fold stats.

The standard ``train_lodo.py`` post-training test eval routes through
``evaluate.evaluate_checkpoint``, which (a) validates the on-disk feature pickle
against a *log-mel* config signature and (b) loads a *global* training-stats
JSON.  Neither holds for Perch features: they are signed with a Perch-specific
payload (no mel keys) and LODO never writes a global stats file — fold stats are
computed in memory and stored inside the checkpoint.  That mismatch crashes the
Perch test eval with "Feature file does not match the current configuration".

This driver sidesteps both problems for *any* model type by normalising the test
split with the exact per-fold ``feature_mean``/``feature_std`` saved in the
checkpoint, then reusing the same ``evaluate_model`` +
``append_official_metrics`` path as the rest of the pipeline.  Outputs are
written in the standard location/format so downstream tooling is unchanged.

Usage
-----
    python scripts/eval_lodo_test_from_ckpt.py \
        data/Development_data/outputs/LODO_D1_*_perch_lodo/model/model_best.pth ...
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import append_official_metrics, load_unseen_domain_by_species
from framework.dataset import MosquitoFeatureDataset, pad_collate_fn
from framework.engine import evaluate_model
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.utilization import (
    build_model, choose_device, make_loader, split_feature_path,
)
from train_lodo import save_json, save_prediction_rows


def evaluate_one(ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    fold = ckpt["fold"]
    device = choose_device(config["device"])

    # Per-fold normalisation stats captured at training time.
    feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(ckpt["feature_std"], dtype=np.float32)

    # Build the test dataset without touching the (mel) signature / global-stats
    # machinery, then inject the checkpoint's per-fold stats directly.
    per_domain = config.get("per_domain_norm", False)
    dataset = MosquitoFeatureDataset(
        feature_pickle_path=split_feature_path(config, "test"),
        feature_stats_path=None,
        max_eval_frames=config.get("max_eval_frames", None),
        training=False,
        normalize_features=False,
        clip_normalize=config.get("clip_normalize", False) or False,
        per_domain_norm=per_domain,
    )
    # per_domain_norm computes per-domain stats from the test split itself
    # (transductive), matching training; otherwise inject the global per-fold stats.
    if not per_domain and config["normalize_features"]:
        dataset.feature_mean = feature_mean
        dataset.feature_std = feature_std
        dataset.normalize_features = True

    eval_batch_size = config.get("eval_batch_size", config["batch_size"])
    loader = make_loader(dataset, eval_batch_size, False,
                         config["num_workers"], device, pad_collate_fn)

    model = build_model(config, device)
    model.load_state_dict(ckpt["model_state_dict"])

    result = evaluate_model(
        model=model,
        dataloader=loader,
        device=device,
        num_species_classes=len(SPECIES_NAMES),
        num_domain_classes=len(DOMAIN_NAMES),
        species_names=SPECIES_NAMES,
        domain_names=DOMAIN_NAMES,
        return_predictions=True,
    )
    official = append_official_metrics(
        result["metrics"], result["predictions"],
        load_unseen_domain_by_species(config), lodo_held_out_domain=fold,
    )
    metrics = official["metrics"]
    predictions = official["predictions"]
    metrics["split"] = "test"
    metrics["checkpoint_path"] = str(ckpt_path)

    out_dir = ckpt_path.parent.parent / "best_model_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "test_metrics.json", metrics)
    save_prediction_rows(out_dir / "test_predictions.jsonl", predictions)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="model_best.pth paths")
    args = ap.parse_args()

    for ckpt in args.checkpoints:
        ckpt_path = Path(ckpt)
        m = evaluate_one(ckpt_path)
        print(f"{ckpt_path.parent.parent.name}: fold={m.get('split')} "
              f"BA_unseen={m.get('BA_unseen')} BA_seen={m.get('BA_seen')} "
              f"DSG={m.get('DSG')} "
              f"species_BA={m.get('species_balanced_accuracy'):.4f}")


if __name__ == "__main__":
    main()
