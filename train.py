"""Training entry point for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.optim import AdamW

from framework.augmentation import build_augmentation_pipeline, build_fbs_mix_fn, mixup_batch
from framework.gradient_reversal import dann_lambda
from framework.utilization import make_balanced_sampler, get_domain_labels
from framework.config import config_signature, feature_signature_payload, load_config, run_context_payload
from framework.dataset import MosquitoFeatureDataset, pad_collate_fn
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
    save_json,
    set_seed,
    split_feature_path,
    training_stats_path,
    max_train_frames,
    release_experiment_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train mosquito classifier.")
    parser.add_argument("--config", type=str, default="configs/default_experiment.json")
    parser.add_argument("--seed", type=int, default=None, help="Override seed in config.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def experiment_name_for_seed(seed: int, config: dict) -> str:
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    min_epoch = int(config.get("early_stopping_min_epoch", 10))
    patience = int(config.get("early_stopping_patience", 10))
    tag = config.get("experiment_tag", "")
    suffix = f"_{tag}" if tag else ""
    return f"MTRCNN_seed{seed}_B{batch_size}_E{epochs}_earlystop_min{min_epoch}_pati{patience}{suffix}"


def evaluate_and_save_outputs(config: dict, checkpoint_path: Path, output_dir: Path, model_name: str) -> dict:
    from evaluate import evaluate_checkpoint, save_prediction_rows

    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    validation_result = evaluate_checkpoint(config, checkpoint_path, "validation", return_predictions=True)
    save_json(model_output_dir / "validation_metrics.json", validation_result["metrics"])
    save_prediction_rows(model_output_dir / "validation_predictions.jsonl", validation_result["predictions"])

    test_result = evaluate_checkpoint(config, checkpoint_path, "test", return_predictions=True)
    save_json(model_output_dir / "test_metrics.json", test_result["metrics"])
    save_prediction_rows(model_output_dir / "test_predictions.jsonl", test_result["predictions"])

    return {
        "output_dir": str(model_output_dir),
        "validation_metrics": validation_result["metrics"],
        "test_metrics": test_result["metrics"],
        "validation_metrics_path": str(model_output_dir / "validation_metrics.json"),
        "validation_predictions_path": str(model_output_dir / "validation_predictions.jsonl"),
        "test_metrics_path": str(model_output_dir / "test_metrics.json"),
        "test_predictions_path": str(model_output_dir / "test_predictions.jsonl"),
    }


def train_experiment(config: dict, overwrite: bool = False) -> dict:
    config = deepcopy(config)
    config["experiment_name"] = experiment_name_for_seed(config["seed"], config)
    set_seed(config["seed"])
    device = choose_device(config["device"])
    output_dir = make_output_dir(config["output_root"], config["experiment_name"])
    model_dir = output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    run_context_path = output_dir / "run_context.json"
    best_checkpoint_path = model_dir / "model_best.pth"
    final_checkpoint_path = model_dir / "model_final.pth"
    best_eval_dir = output_dir / "best_model_eval"
    final_eval_dir = output_dir / "final_model_eval"
    required_eval_files = [
        best_eval_dir / "validation_metrics.json",
        best_eval_dir / "validation_predictions.jsonl",
        best_eval_dir / "test_metrics.json",
        best_eval_dir / "test_predictions.jsonl",
        final_eval_dir / "validation_metrics.json",
        final_eval_dir / "validation_predictions.jsonl",
        final_eval_dir / "test_metrics.json",
        final_eval_dir / "test_predictions.jsonl",
    ]
    current_run_context = run_context_payload(config)

    if (
        run_context_path.exists()
        and best_checkpoint_path.exists()
        and final_checkpoint_path.exists()
        and all(path.exists() for path in required_eval_files)
        and load_json(run_context_path) == current_run_context
        and not overwrite
    ):
        print(f"loading from {best_checkpoint_path}")
        best_validation_metrics = load_json(best_eval_dir / "validation_metrics.json")
        best_test_metrics = load_json(best_eval_dir / "test_metrics.json")
        final_validation_metrics = load_json(final_eval_dir / "validation_metrics.json")
        final_test_metrics = load_json(final_eval_dir / "test_metrics.json")
        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "best_eval": {
                "output_dir": str(best_eval_dir),
                "validation_metrics": best_validation_metrics,
                "test_metrics": best_test_metrics,
                "validation_metrics_path": str(best_eval_dir / "validation_metrics.json"),
                "validation_predictions_path": str(best_eval_dir / "validation_predictions.jsonl"),
                "test_metrics_path": str(best_eval_dir / "test_metrics.json"),
                "test_predictions_path": str(best_eval_dir / "test_predictions.jsonl"),
            },
            "final_eval": {
                "output_dir": str(final_eval_dir),
                "validation_metrics": final_validation_metrics,
                "test_metrics": final_test_metrics,
                "validation_metrics_path": str(final_eval_dir / "validation_metrics.json"),
                "validation_predictions_path": str(final_eval_dir / "validation_predictions.jsonl"),
                "test_metrics_path": str(final_eval_dir / "test_metrics.json"),
                "test_predictions_path": str(final_eval_dir / "test_predictions.jsonl"),
            },
        }

    lock_path = acquire_experiment_lock(output_dir, config["experiment_name"])
    if lock_path is None:
        print(f"already running: {output_dir / '.experiment.lock'}")
        return {
            "status": "running",
            "output_dir": str(output_dir),
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
        }
    print(f"training model to {output_dir}")
    try:
        save_json(run_context_path, current_run_context)
        save_json(output_dir / "resolved_config.json", config)
        logger = make_logger(output_dir / "train.log")

        expected_training_feature_signature = config_signature(feature_signature_payload(config, "training"))
        expected_validation_feature_signature = config_signature(feature_signature_payload(config, "validation"))
        expected_training_stats_signature = expected_training_feature_signature

        aug_pipeline = build_augmentation_pipeline(config.get("augmentation"))
        aug_cfg      = config.get("augmentation", {})
        mixup_cfg    = aug_cfg.get("mixup", {})
        mixup_fn     = (
            (lambda f, sp, dom: mixup_batch(f, sp, dom, alpha=mixup_cfg.get("alpha", 0.4)))
            if mixup_cfg.get("enabled", False) else None
        )
        fbs_mix_fn   = build_fbs_mix_fn(config)
        clip_norm    = config.get("clip_normalize", False)

        train_dataset = MosquitoFeatureDataset(
            feature_pickle_path=split_feature_path(config, "training"),
            feature_stats_path=training_stats_path(config),
            max_train_frames=max_train_frames(config),
            training=True,
            normalize_features=config["normalize_features"],
            clip_normalize=clip_norm,
            expected_feature_signature=expected_training_feature_signature,
            expected_stats_signature=expected_training_stats_signature,
            augment=aug_pipeline,
        )
        val_dataset = MosquitoFeatureDataset(
            feature_pickle_path=split_feature_path(config, "validation"),
            feature_stats_path=training_stats_path(config),
            max_train_frames=None,
            training=False,
            normalize_features=config["normalize_features"],
            clip_normalize=clip_norm,
            expected_feature_signature=expected_validation_feature_signature,
            expected_stats_signature=expected_training_stats_signature,
        )
        print(f"loading from {split_feature_path(config, 'training')}")
        print(f"loading from {split_feature_path(config, 'validation')}")
        if config["normalize_features"]:
            print(f"loading from {training_stats_path(config)}")

        sampler = (
            make_balanced_sampler(get_domain_labels(train_dataset))
            if config.get("domain_balanced_sampling", False) else None
        )
        train_loader = make_loader(train_dataset, config["batch_size"], True, config["num_workers"], device, pad_collate_fn, sampler=sampler)
        eval_batch_size = config.get("eval_batch_size", config["batch_size"])
        val_loader = make_loader(val_dataset, eval_batch_size, False, config["num_workers"], device, pad_collate_fn)

        # Species inverse-frequency loss weights (optional)
        species_loss_weight = None
        if config.get("species_balanced_loss", False):
            from collections import Counter
            counts = Counter(item["species_label"] for item in train_dataset.samples)
            w = torch.tensor(
                [1.0 / max(counts.get(i, 1), 1) for i in range(len(SPECIES_NAMES))],
                dtype=torch.float32, device=device,
            )
            species_loss_weight = w / w.sum() * len(SPECIES_NAMES)

        model = build_model(config, device)
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        early_stopping_min_epoch = int(config.get("early_stopping_min_epoch", 10))
        early_stopping_patience = int(config.get("early_stopping_patience", 10))

        best_score = float("-inf")
        best_epoch = 0
        best_val_metrics = {}
        last_val_metrics = {}
        epochs_without_improvement = 0
        use_dann    = config.get("domain_adversarial", False)
        lambda_max  = float(config.get("grl_lambda_max", 1.0))
        total_epochs = int(config["epochs"])

        for epoch in range(1, total_epochs + 1):
            grl_lam = dann_lambda(epoch, total_epochs, lambda_max) if use_dann else None
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                device=device,
                mixup_fn=mixup_fn,
                fbs_mix_fn=fbs_mix_fn,
                grl_lambda=grl_lam,
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
                "epoch": epoch,
                "train_loss": round(train_metrics["loss"], 6),
                "train_species_loss": round(train_metrics["species_loss"], 6),
                "train_domain_loss": round(train_metrics["domain_loss"], 6),
                "train_species_accuracy": round(train_metrics["species_accuracy"], 6),
                "train_domain_accuracy": round(train_metrics["domain_accuracy"], 6),
                "val_loss": round(val_metrics["loss"], 6),
                "val_species_loss": round(val_metrics["species_loss"], 6),
                "val_domain_loss": round(val_metrics["domain_loss"], 6),
                "val_species_accuracy": round(val_metrics["species_accuracy"], 6),
                "val_species_balanced_accuracy": round(val_metrics["species_balanced_accuracy"], 6),
                "val_domain_accuracy": round(val_metrics["domain_accuracy"], 6),
                "val_domain_balanced_accuracy": round(val_metrics["domain_balanced_accuracy"], 6),
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_metrics(output_dir / "metrics.csv", row)
            logger.info(row)

            current_score = val_metrics["species_balanced_accuracy"]
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_val_metrics = dict(val_metrics)
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config,
                        "epoch": epoch,
                        "val_metrics": best_val_metrics,
                        "selection_metric": "species_balanced_accuracy",
                    },
                    best_checkpoint_path,
                )
                logger.info("Saved best checkpoint to %s", best_checkpoint_path)
            elif epoch >= early_stopping_min_epoch:
                epochs_without_improvement += 1

            if epoch >= early_stopping_min_epoch and epochs_without_improvement >= early_stopping_patience:
                logger.info(
                    "Early stopping at epoch %s. Best epoch: %s, best validation species_balanced_accuracy: %.6f, min_epoch: %s, patience: %s",
                    epoch,
                    best_epoch,
                    best_score,
                    early_stopping_min_epoch,
                    early_stopping_patience,
                )
                break

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_metrics": last_val_metrics,
            },
            final_checkpoint_path,
        )
        logger.info("Saved final checkpoint to %s", final_checkpoint_path)

        logger.info("Evaluating best checkpoint outputs.")
        best_eval = evaluate_and_save_outputs(config, best_checkpoint_path, output_dir, "best_model_eval")
        logger.info("Evaluating final checkpoint outputs.")
        final_eval = evaluate_and_save_outputs(config, final_checkpoint_path, output_dir, "final_model_eval")

        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "best_checkpoint_path": str(best_checkpoint_path),
            "final_checkpoint_path": str(final_checkpoint_path),
            "best_eval": best_eval,
            "final_eval": final_eval,
        }
    finally:
        release_experiment_lock(lock_path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    result = train_experiment(config, overwrite=args.overwrite)
    if result["status"] == "running":
        return


if __name__ == "__main__":
    main()
