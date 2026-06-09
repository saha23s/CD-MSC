"""Leave-One-Domain-Out cross-validation training for the DCASE2026 mosquito baseline.

Each fold holds out one domain (D1–D5) as the validation set and trains on the
remaining four domains. Features are loaded from the pre-extracted training and
validation pickles and split in memory — no re-extraction required.

Usage:
    python train_lodo.py --fold D3
    python train_lodo.py --fold D3 --seed 3407
    python train_lodo.py --fold D3 --config configs/default_experiment.json --overwrite

Author: Sulagna Saha
"""

import argparse
import hashlib
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW

from evaluate import evaluate_checkpoint, save_prediction_rows
from framework.augmentation import build_augmentation_pipeline, build_fbs_mix_fn, mixup_batch
from framework.config import config_signature, load_config
from framework.gradient_reversal import dann_lambda
from framework.utilization import make_balanced_sampler, get_domain_labels
from framework.dataset import LodoFeatureDataset, pad_collate_fn
from framework.engine import evaluate_model, train_one_epoch
from framework.metadata import DOMAIN_NAMES, SPECIES_NAMES
from framework.utilization import (
    acquire_experiment_lock,
    append_metrics,
    build_model,
    choose_device,
    load_json,
    make_loader,
    make_logger,
    make_output_dir,
    release_experiment_lock,
    save_json,
    set_seed,
    split_feature_path,
    max_train_frames,
)


# ---------------------------------------------------------------------------
# LODO split logic
# ---------------------------------------------------------------------------

def get_lodo_folds(items: List[Dict], held_out_domain: str) -> Tuple[List[Dict], List[Dict]]:
    """Split items into train / val by leaving one domain out.

    Args:
        items: list of feature-payload dicts, each with a ``domain`` key.
        held_out_domain: domain name to hold out for validation (e.g. ``"D3"``).

    Returns:
        (train_items, val_items)
    """
    train_items = [item for item in items if item["domain"] != held_out_domain]
    val_items   = [item for item in items if item["domain"] == held_out_domain]
    return train_items, val_items


def compute_lodo_stats(train_items: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-mel-bin mean and std from the LODO training fold."""
    feature_sum:    Optional[np.ndarray] = None
    feature_sq_sum: Optional[np.ndarray] = None
    total_frames = 0

    for item in train_items:
        feature = item["feature"].astype(np.float64)           # [T, n_mels]
        if feature_sum is None:
            feature_sum    = feature.sum(axis=0)
            feature_sq_sum = np.square(feature).sum(axis=0)
        else:
            feature_sum    += feature.sum(axis=0)
            feature_sq_sum += np.square(feature).sum(axis=0)
        total_frames += feature.shape[0]

    mean     = (feature_sum / total_frames).astype(np.float32)
    variance = np.maximum(feature_sq_sum / total_frames - np.square(mean.astype(np.float64)), 1e-12)
    std      = np.sqrt(variance).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pickle_items(path: Path) -> Tuple[List[Dict], str]:
    """Load items and config_signature from a feature pickle."""
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return payload["items"], payload.get("config_signature", "")


def lodo_run_context(config: Dict, fold: str, train_sig: str, val_sig: str) -> Dict:
    combined = hashlib.sha256(f"{train_sig}|{val_sig}".encode()).hexdigest()
    return {
        "resolved_config_signature": config_signature(config),
        "fold": fold,
        "trainval_feature_signature": combined,
    }


def experiment_name(fold: str, seed: int, config: Dict) -> str:
    batch_size = int(config["batch_size"])
    epochs     = int(config["epochs"])
    min_epoch  = int(config.get("early_stopping_min_epoch", 10))
    patience   = int(config.get("early_stopping_patience", 10))
    base = f"LODO_{fold}_seed{seed}_B{batch_size}_E{epochs}_earlystop_min{min_epoch}_pati{patience}"
    tag = config.get("experiment_tag", "")
    return f"{base}_{tag}" if tag else base


def evaluate_and_save_lodo_val(
    model,
    val_loader,
    device,
    output_dir: Path,
    checkpoint_path: Path,
    fold: str,
) -> Dict:
    """Evaluate model on the held-out LODO validation domain and persist results."""
    result = evaluate_model(
        model=model,
        dataloader=val_loader,
        device=device,
        num_species_classes=len(SPECIES_NAMES),
        num_domain_classes=len(DOMAIN_NAMES),
        species_names=SPECIES_NAMES,
        domain_names=DOMAIN_NAMES,
        return_predictions=True,
    )
    metrics     = result["metrics"]
    predictions = result["predictions"]
    metrics["split"]           = f"lodo_val_{fold}"
    metrics["checkpoint_path"] = str(checkpoint_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "lodo_val_metrics.json", metrics)
    save_prediction_rows(output_dir / "lodo_val_predictions.jsonl", predictions)
    return metrics


def evaluate_and_save_test(
    config: Dict,
    checkpoint_path: Path,
    output_dir: Path,
    lodo_held_out_domain: Optional[str] = None,
) -> Dict:
    """Evaluate checkpoint on the official test split and persist results."""
    result = evaluate_checkpoint(
        config, checkpoint_path, "test", return_predictions=True,
        lodo_held_out_domain=lodo_held_out_domain,
    )
    metrics     = result["metrics"]
    predictions = result["predictions"]

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "test_metrics.json", metrics)
    save_prediction_rows(output_dir / "test_predictions.jsonl", predictions)
    return metrics


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_lodo_experiment(config: Dict, fold: str, overwrite: bool = False) -> Dict:
    config = deepcopy(config)
    seed   = int(config["seed"])
    set_seed(seed)
    device = choose_device(config["device"])

    exp_name   = experiment_name(fold, seed, config)
    output_dir = make_output_dir(config["output_root"], exp_name)
    model_dir  = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path  = model_dir / "model_best.pth"
    final_ckpt_path = model_dir / "model_final.pth"
    run_context_path = output_dir / "run_context.json"

    # ---- load feature pickles -----------------------------------------------
    train_pkl_path = split_feature_path(config, "training")
    val_pkl_path   = split_feature_path(config, "validation")
    print(f"loading from {train_pkl_path}")
    print(f"loading from {val_pkl_path}")

    train_all_items, train_sig = load_pickle_items(train_pkl_path)
    val_all_items,   val_sig   = load_pickle_items(val_pkl_path)
    all_items = train_all_items + val_all_items

    current_context = lodo_run_context(config, fold, train_sig, val_sig)

    # ---- resumability check -------------------------------------------------
    required_eval_files = [
        output_dir / "best_model_eval"  / "lodo_val_metrics.json",
        output_dir / "best_model_eval"  / "test_metrics.json",
        output_dir / "final_model_eval" / "lodo_val_metrics.json",
        output_dir / "final_model_eval" / "test_metrics.json",
    ]
    if (
        run_context_path.exists()
        and best_ckpt_path.exists()
        and final_ckpt_path.exists()
        and all(p.exists() for p in required_eval_files)
        and load_json(run_context_path) == current_context
        and not overwrite
    ):
        print(f"already completed — loading results from {output_dir}")
        return {
            "status":             "completed",
            "output_dir":         str(output_dir),
            "best_checkpoint":    str(best_ckpt_path),
            "final_checkpoint":   str(final_ckpt_path),
        }

    lock_path = acquire_experiment_lock(output_dir, exp_name)
    if lock_path is None:
        print(f"already running: {output_dir / '.experiment.lock'}")
        return {"status": "running", "output_dir": str(output_dir)}

    print(f"training LODO fold={fold} seed={seed} → {output_dir}")
    try:
        save_json(run_context_path, current_context)
        save_json(output_dir / "resolved_config.json", config)
        logger = make_logger(output_dir / "train.log")

        # ---- LODO split -----------------------------------------------------
        train_items, val_items = get_lodo_folds(all_items, fold)
        logger.info(
            "LODO fold=%s | train=%d samples (domains %s) | val=%d samples",
            fold,
            len(train_items),
            sorted({item["domain"] for item in train_items}),
            len(val_items),
        )

        # ---- feature stats from the LODO training fold ----------------------
        feature_mean, feature_std = compute_lodo_stats(train_items)

        # ---- augmentation ---------------------------------------------------
        aug_pipeline = build_augmentation_pipeline(config.get("augmentation"))
        aug_cfg      = config.get("augmentation", {})
        mixup_cfg    = aug_cfg.get("mixup", {})
        mixup_fn     = (
            (lambda f, sp, dom: mixup_batch(f, sp, dom, alpha=mixup_cfg.get("alpha", 0.4)))
            if mixup_cfg.get("enabled", False) else None
        )
        fbs_mix_fn   = build_fbs_mix_fn(config)
        if aug_pipeline:
            logger.info("Augmentation pipeline: %s", aug_pipeline)
        if mixup_cfg.get("enabled", False):
            logger.info("Mixup enabled with alpha=%.2f", mixup_cfg.get("alpha", 0.4))
        if fbs_mix_fn:
            logger.info("FBS-Mix enabled: mix bins 0-%d, protect bins %d-%d",
                        config.get("fbmix_species_lo", 9) - 1,
                        config.get("fbmix_species_lo", 9),
                        config.get("fbmix_species_hi", 36) - 1)

        # ---- datasets & loaders ---------------------------------------------
        n_train_frames = max_train_frames(config)
        clip_norm      = config.get("clip_normalize", False)
        max_eval_fr    = config.get("max_eval_frames", None)
        train_dataset  = LodoFeatureDataset(
            items=train_items,
            feature_mean=feature_mean,
            feature_std=feature_std,
            max_train_frames=n_train_frames,
            training=True,
            normalize_features=config["normalize_features"],
            clip_normalize=clip_norm,
            augment=aug_pipeline,
        )
        val_dataset = LodoFeatureDataset(
            items=val_items,
            feature_mean=feature_mean,
            feature_std=feature_std,
            max_train_frames=None,
            max_eval_frames=max_eval_fr,
            training=False,
            normalize_features=config["normalize_features"],
            clip_normalize=clip_norm,
        )
        sampler = (
            make_balanced_sampler(get_domain_labels(train_dataset))
            if config.get("domain_balanced_sampling", False) else None
        )
        train_loader = make_loader(train_dataset, config["batch_size"], True, config["num_workers"], device, pad_collate_fn, sampler=sampler)
        val_loader   = make_loader(val_dataset,   config.get("eval_batch_size",
                                                             config["batch_size"]), False, config["num_workers"], device, pad_collate_fn)

        # ---- species loss weight (inverse-frequency, optional) ---------------
        species_loss_weight = None
        if config.get("species_balanced_loss", False):
            from collections import Counter
            counts = Counter(item["species_label"] for item in train_dataset.samples)
            w = torch.tensor(
                [1.0 / max(counts.get(i, 1), 1) for i in range(len(SPECIES_NAMES))],
                dtype=torch.float32, device=device,
            )
            species_loss_weight = w / w.sum() * len(SPECIES_NAMES)  # normalise

        # ---- model + optimiser ----------------------------------------------
        model     = build_model(config, device)
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

        early_stopping_min_epoch = int(config.get("early_stopping_min_epoch", 10))
        early_stopping_patience  = int(config.get("early_stopping_patience", 10))

        best_score               = float("-inf")
        best_epoch               = 0
        best_val_metrics: Dict   = {}
        last_val_metrics:  Dict  = {}
        epochs_without_improvement = 0

        # ---- training loop --------------------------------------------------
        use_dann     = config.get("domain_adversarial", False)
        lambda_max   = float(config.get("grl_lambda_max", 1.0))
        total_epochs = int(config["epochs"])

        for epoch in range(1, total_epochs + 1):
            grl_lam = dann_lambda(epoch, total_epochs, lambda_max) if use_dann else None
            train_metrics = train_one_epoch(
                model=model, dataloader=train_loader, optimizer=optimizer, device=device,
                mixup_fn=mixup_fn, fbs_mix_fn=fbs_mix_fn, grl_lambda=grl_lam,
                domain_loss_weight=config.get("domain_loss_weight", 1.0),
                scol_weight=config.get("scol_weight", 0.0),
                scol_tau=config.get("scol_tau", 0.01),
                dicl_weight=config.get("dicl_weight", 0.0),
                dicl_tau=config.get("dicl_tau", 0.07),
                sdal_weight=config.get("sdal_weight", 0.0),
                sdal_sigma=config.get("sdal_sigma", 1.0),
                species_loss_weight=species_loss_weight,
            )
            val_metrics = evaluate_model(
                model=model,
                dataloader=val_loader,
                device=device,
                num_species_classes=len(SPECIES_NAMES),
                num_domain_classes=len(DOMAIN_NAMES),
            )
            last_val_metrics = val_metrics

            row = {
                "epoch":                       epoch,
                "train_loss":                  round(train_metrics["loss"], 6),
                "train_species_loss":          round(train_metrics["species_loss"], 6),
                "train_domain_loss":           round(train_metrics["domain_loss"], 6),
                "train_species_accuracy":      round(train_metrics["species_accuracy"], 6),
                "train_domain_accuracy":       round(train_metrics["domain_accuracy"], 6),
                "val_loss":                    round(val_metrics["loss"], 6),
                "val_species_loss":            round(val_metrics["species_loss"], 6),
                "val_domain_loss":             round(val_metrics["domain_loss"], 6),
                "val_species_accuracy":        round(val_metrics["species_accuracy"], 6),
                "val_species_balanced_accuracy": round(val_metrics["species_balanced_accuracy"], 6),
                "val_domain_accuracy":         round(val_metrics["domain_accuracy"], 6),
                "val_domain_balanced_accuracy": round(val_metrics["domain_balanced_accuracy"], 6),
                "lr":                          optimizer.param_groups[0]["lr"],
            }
            append_metrics(output_dir / "metrics.csv", row)
            logger.info(row)

            current_score = val_metrics["species_balanced_accuracy"]
            if current_score > best_score:
                best_score       = current_score
                best_epoch       = epoch
                best_val_metrics = dict(val_metrics)
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config":           config,
                        "epoch":            epoch,
                        "fold":             fold,
                        "val_metrics":      best_val_metrics,
                        "selection_metric": "species_balanced_accuracy",
                        "feature_mean":     feature_mean,
                        "feature_std":      feature_std,
                    },
                    best_ckpt_path,
                )
                logger.info("Saved best checkpoint → %s", best_ckpt_path)
            elif epoch >= early_stopping_min_epoch:
                epochs_without_improvement += 1

            if epoch >= early_stopping_min_epoch and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %d. Best epoch: %d, best lodo_val BA=%.6f",
                    epoch, best_epoch, best_score,
                )
                break

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config":           config,
                "epoch":            epoch,
                "fold":             fold,
                "val_metrics":      last_val_metrics,
                "feature_mean":     feature_mean,
                "feature_std":      feature_std,
            },
            final_ckpt_path,
        )
        logger.info("Saved final checkpoint → %s", final_ckpt_path)

        # ---- post-training evaluation: best checkpoint ----------------------
        logger.info("Evaluating best checkpoint (LODO val + test).")
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
        evaluate_and_save_lodo_val(model, val_loader, device, output_dir / "best_model_eval", best_ckpt_path, fold)
        evaluate_and_save_test(config, best_ckpt_path, output_dir / "best_model_eval", lodo_held_out_domain=fold)

        # ---- post-training evaluation: final checkpoint ---------------------
        logger.info("Evaluating final checkpoint (LODO val + test).")
        model.load_state_dict(torch.load(final_ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
        evaluate_and_save_lodo_val(model, val_loader, device, output_dir / "final_model_eval", final_ckpt_path, fold)
        evaluate_and_save_test(config, final_ckpt_path, output_dir / "final_model_eval", lodo_held_out_domain=fold)

        return {
            "status":           "completed",
            "output_dir":       str(output_dir),
            "best_checkpoint":  str(best_ckpt_path),
            "final_checkpoint": str(final_ckpt_path),
        }

    finally:
        release_experiment_lock(lock_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LODO fold for the DCASE2026 mosquito baseline."
    )
    parser.add_argument("--config",    type=str, default="configs/default_experiment.json")
    parser.add_argument(
        "--fold",
        type=str,
        required=True,
        choices=DOMAIN_NAMES,
        help="Domain to hold out as the validation set (e.g. D3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in the config.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args   = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    train_lodo_experiment(config, fold=args.fold, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
