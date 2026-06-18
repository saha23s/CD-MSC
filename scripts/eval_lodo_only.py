#!/usr/bin/env python3
"""Re-run eval ONLY on finished LODO runs whose post-training eval was lost.

A permission bug on ``metrics.csv`` crashed many runs *after* training completed
(both ``model_best.pth`` and ``model_final.pth`` saved) but *before* the eval
artifacts were written. Re-launching ``train_lodo.py`` would retrain from
scratch; this script instead replays exactly the post-training eval block of
``train_lodo.train_lodo_experiment`` (lines that call ``evaluate_and_save_*``)
for an existing run directory, using the feature stats saved inside the
checkpoint so results match what training would have produced.

Usage (from repo root, .venv active):
    python scripts/eval_lodo_only.py --run-dir data/Development_data/outputs/<run> [<run> ...]
    python scripts/eval_lodo_only.py --run-dir <run> --which best   # skip final
"""
import argparse
import re
import sys
from pathlib import Path
from typing import List

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_lodo as T
from framework.dataset import LodoFeatureDataset, pad_collate_fn
from framework.utilization import build_model, choose_device, make_loader

_FOLD_RE = re.compile(r"LODO_(D\d)_")


def fold_from_name(run_dir: Path) -> str:
    m = _FOLD_RE.search(run_dir.name)
    if not m:
        raise ValueError(f"Cannot parse fold from run dir name: {run_dir.name}")
    return m.group(1)


_PICKLE_CACHE: dict = {}  # path -> items; the training/val pickles are global (4 GB),
                          # so load once and reuse across all runs in a batch.


def _load_items_cached(path) -> list:
    key = str(path)
    if key not in _PICKLE_CACHE:
        _PICKLE_CACHE[key], _ = T.load_pickle_items(path)
    return _PICKLE_CACHE[key]


def build_val_loader(config: dict, fold: str, ckpt: dict, device: torch.device):
    """Held-out-domain val loader, mirroring train_lodo dataset construction."""
    train_items = _load_items_cached(T.split_feature_path(config, "training"))
    val_items = _load_items_cached(T.split_feature_path(config, "validation"))
    _, held_out_items = T.get_lodo_folds(train_items + val_items, fold)

    val_dataset = LodoFeatureDataset(
        items=held_out_items,
        feature_mean=ckpt["feature_mean"],
        feature_std=ckpt["feature_std"],
        max_train_frames=None,
        max_eval_frames=config.get("max_eval_frames", None),
        training=False,
        normalize_features=config["normalize_features"],
        clip_normalize=config.get("clip_normalize", False),
    )
    return make_loader(
        val_dataset,
        config.get("eval_batch_size", config["batch_size"]),
        False,
        config["num_workers"],
        device,
        pad_collate_fn,
    )


def eval_one(run_dir: Path, which: str, device: torch.device) -> None:
    config = T.load_json(run_dir / "resolved_config.json")
    fold = fold_from_name(run_dir)
    model = build_model(config, device)

    tags = {"best": "model_best.pth", "final": "model_final.pth"}
    if which != "both":
        tags = {which: tags[which]}

    val_loader = None
    for tag, ckpt_name in tags.items():
        ckpt_path = run_dir / "model" / ckpt_name
        if not ckpt_path.exists():
            print(f"  [{run_dir.name}] missing {ckpt_name}, skipping {tag}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if val_loader is None:  # built once; stats identical across best/final
            val_loader = build_val_loader(config, fold, ckpt, device)
        model.load_state_dict(ckpt["model_state_dict"])
        out_dir = run_dir / f"{tag}_model_eval"
        T.evaluate_and_save_lodo_val(model, val_loader, device, out_dir, ckpt_path, fold)
        T.evaluate_and_save_test(config, ckpt_path, out_dir, lodo_held_out_domain=fold)
        print(f"  [{run_dir.name}] {tag}: wrote {out_dir.name}/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=str, nargs="+", required=True,
                   help="One or more LODO run output directories.")
    p.add_argument("--which", choices=["best", "final", "both"], default="both")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    print(f"device: {device}")
    run_dirs: List[Path] = [Path(d) for d in args.run_dir]
    for rd in run_dirs:
        if not rd.exists():
            print(f"  SKIP (missing): {rd}")
            continue
        print(f"evaluating {rd.name} (fold {fold_from_name(rd)})")
        try:
            eval_one(rd, args.which, device)
        except Exception as exc:  # keep the batch going if one run is corrupt
            print(f"  ERROR on {rd.name}: {type(exc).__name__}: {exc}")
    print("done.")


if __name__ == "__main__":
    main()
