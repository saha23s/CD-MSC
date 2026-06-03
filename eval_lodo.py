"""Evaluate saved LODO checkpoints without re-training.

Loads existing best/final checkpoints from a completed (or interrupted) LODO
training run and runs the LODO-val + test evaluation, writing results to the
same output directory structure as train_lodo.py.

Usage:
    python eval_lodo.py --fold D1
    python eval_lodo.py --fold all
    python eval_lodo.py --fold D3 --config configs/default_experiment.json --overwrite
"""

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Dict

import torch

from train_lodo import (
    compute_lodo_stats,
    evaluate_and_save_lodo_val,
    evaluate_and_save_test,
    experiment_name,
    get_lodo_folds,
    load_pickle_items,
)
from framework.config import load_config
from framework.dataset import LodoFeatureDataset, pad_collate_fn
from framework.metadata import DOMAIN_NAMES
from framework.utilization import (
    build_model,
    choose_device,
    make_loader,
    set_seed,
    split_feature_path,
)

FOLDS = ["D1", "D2", "D3", "D4", "D5"]


def eval_fold(
    config: Dict, fold: str, overwrite: bool,
    ttbn: bool = False, tent_steps: int = 0, tent_lr: float = 1e-3,
) -> None:
    config = deepcopy(config)
    if ttbn:
        config["ttbn"] = True
    seed   = int(config["seed"])
    set_seed(seed)
    device = choose_device(config["device"])

    exp_name   = experiment_name(fold, seed, config)
    output_dir = Path(config["output_root"]) / exp_name
    model_dir  = output_dir / "model"

    best_ckpt_path  = model_dir / "model_best.pth"
    final_ckpt_path = model_dir / "model_final.pth"

    for ckpt in (best_ckpt_path, final_ckpt_path):
        if not ckpt.exists():
            print(f"[{fold}] checkpoint not found: {ckpt} — skipping fold")
            return

    if tent_steps > 0:
        eval_suffix = f"_tent{tent_steps}"
    elif ttbn:
        eval_suffix = "_ttbn"
    else:
        eval_suffix = ""
    best_eval_dir  = output_dir / f"best_model_eval{eval_suffix}"
    final_eval_dir = output_dir / f"final_model_eval{eval_suffix}"

    required_eval_files = [
        best_eval_dir  / "lodo_val_metrics.json",
        best_eval_dir  / "test_metrics.json",
        final_eval_dir / "lodo_val_metrics.json",
        final_eval_dir / "test_metrics.json",
    ]
    if not overwrite and all(p.exists() for p in required_eval_files):
        print(f"[{fold}] already evaluated — skipping (use --overwrite to redo)")
        return

    # ---- reconstruct val loader ---------------------------------------------
    train_pkl_path = split_feature_path(config, "training")
    val_pkl_path   = split_feature_path(config, "validation")
    train_all_items, _ = load_pickle_items(train_pkl_path)
    val_all_items,   _ = load_pickle_items(val_pkl_path)
    all_items = train_all_items + val_all_items

    train_items, val_items = get_lodo_folds(all_items, fold)
    feature_mean, feature_std = compute_lodo_stats(train_items)

    val_dataset = LodoFeatureDataset(
        items=val_items,
        feature_mean=feature_mean,
        feature_std=feature_std,
        max_train_frames=None,
        max_eval_frames=config.get("max_eval_frames", None),
        training=False,
        normalize_features=config["normalize_features"],
        clip_normalize=config.get("clip_normalize", False),
    )
    val_loader = make_loader(
        val_dataset,
        config.get("eval_batch_size", config["batch_size"]),
        False,
        config["num_workers"],
        device,
        pad_collate_fn,
    )

    model = build_model(config, device)

    # ---- best checkpoint ----------------------------------------------------
    print(f"[{fold}] evaluating best checkpoint …")
    model.load_state_dict(
        torch.load(best_ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    evaluate_and_save_lodo_val(model, val_loader, device, best_eval_dir, best_ckpt_path, fold)
    evaluate_and_save_test(config, best_ckpt_path, best_eval_dir,
                           lodo_held_out_domain=fold, tent_steps=tent_steps, tent_lr=tent_lr)

    # ---- final checkpoint ---------------------------------------------------
    print(f"[{fold}] evaluating final checkpoint …")
    model.load_state_dict(
        torch.load(final_ckpt_path, map_location=device, weights_only=False)["model_state_dict"]
    )
    evaluate_and_save_lodo_val(model, val_loader, device, final_eval_dir, final_ckpt_path, fold)
    evaluate_and_save_test(config, final_ckpt_path, final_eval_dir,
                           lodo_held_out_domain=fold, tent_steps=tent_steps, tent_lr=tent_lr)

    print(f"[{fold}] done → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved LODO checkpoints.")
    parser.add_argument("--fold", default="all", help="Fold to evaluate: D1–D5 or 'all'")
    parser.add_argument("--config", default="configs/default_experiment.json")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate even if results exist")
    parser.add_argument("--ttbn", action="store_true", help="Enable test-time batch norm; writes to *_ttbn/ subdirs")
    parser.add_argument("--tent-steps", type=int, default=0, help="TENT adaptation steps per batch (0=off); writes to *_tent{n}/ subdirs")
    parser.add_argument("--tent-lr", type=float, default=1e-3, help="TENT Adam learning rate")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Override eval_batch_size in config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.eval_batch_size is not None:
        config["eval_batch_size"] = args.eval_batch_size
    folds  = FOLDS if args.fold == "all" else [args.fold]

    for fold in folds:
        eval_fold(config, fold, overwrite=args.overwrite, ttbn=args.ttbn,
                  tent_steps=args.tent_steps, tent_lr=args.tent_lr)


if __name__ == "__main__":
    main()
