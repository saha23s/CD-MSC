"""Training and evaluation utilities for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from framework.losses import domain_invariant_contrastive_loss, species_cohesion_contrastive_loss, species_conditional_mmd_loss
from framework.metadata import SPECIES_WINGBEAT_TARGETS
from framework.group_dro import GroupDROState


def balanced_accuracy(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    """Mean recall over classes present in the evaluation set."""
    recalls = []
    for class_index in range(num_classes):
        class_mask = labels == class_index
        if class_mask.any():
            class_recall = (preds[class_mask] == class_index).float().mean().item()
            recalls.append(class_recall)
    return float(np.mean(recalls)) if recalls else 0.0


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    return float((preds == labels).float().mean().item())


def evaluate_model(
    model,
    dataloader,
    device,
    num_species_classes: int,
    num_domain_classes: int,
    species_names=None,
    domain_names=None,
    return_predictions: bool = False,
    ttbn: bool = False,
    tent_steps: int = 0,
    tent_lr: float = 1e-3,
) -> dict:
    """Evaluate model on a dataloader.

    Args:
        ttbn:       Test-Time Batch Normalisation — BN layers use batch stats.
        tent_steps: TENT adaptation steps per batch (Wang et al. 2021).
                    Adapts BN/LN affine params via entropy minimisation before
                    each prediction. 0 = disabled. Mutually exclusive with ttbn.
        tent_lr:    Learning rate for TENT Adam optimiser.
    """
    tent_optimizer = None
    if tent_steps > 0:
        from framework.tent import configure_tent
        tent_optimizer = configure_tent(model, lr=tent_lr)
        # configure_tent already sets model.train() with only norm affines unfrozen
    else:
        model.eval()
        if ttbn:
            for m in model.modules():
                if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                    m.training = True
    species_preds = []
    species_labels = []
    domain_preds = []
    domain_labels = []
    batch_domain_names = []
    prediction_rows = []
    total_loss = 0.0
    total_species_loss = 0.0
    total_domain_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            lengths = batch["lengths"].to(device)
            batch_species_labels = batch["species_labels"].to(device)
            batch_domain_labels = batch["domain_labels"].to(device)

            if tent_optimizer is not None:
                from framework.tent import tent_step
                tent_step(model, features, lengths, tent_optimizer, n_steps=tent_steps)

            outputs = model(features, lengths)
            batch_species_logits = outputs["species_logits"]
            batch_domain_logits = outputs["domain_logits"]

            batch_species_loss = F.cross_entropy(batch_species_logits, batch_species_labels)
            batch_domain_loss = F.cross_entropy(batch_domain_logits, batch_domain_labels)
            batch_loss = batch_species_loss + batch_domain_loss

            batch_species_preds = torch.argmax(batch_species_logits, dim=1)
            batch_domain_preds = torch.argmax(batch_domain_logits, dim=1)
            batch_species_probs = torch.softmax(batch_species_logits, dim=1)

            batch_size = batch_species_labels.size(0)
            total_loss += batch_loss.item() * batch_size
            total_species_loss += batch_species_loss.item() * batch_size
            total_domain_loss += batch_domain_loss.item() * batch_size
            total_items += batch_size

            species_preds.append(batch_species_preds.cpu())
            species_labels.append(batch_species_labels.cpu())
            domain_preds.append(batch_domain_preds.cpu())
            domain_labels.append(batch_domain_labels.cpu())
            batch_domain_names.extend(batch["domain"])

            if return_predictions:
                species_logits_cpu = batch_species_logits.detach().cpu()
                species_probs_cpu = batch_species_probs.detach().cpu()
                species_labels_cpu = batch_species_labels.detach().cpu()
                domain_labels_cpu = batch_domain_labels.detach().cpu()
                species_preds_cpu = batch_species_preds.detach().cpu()
                domain_preds_cpu = batch_domain_preds.detach().cpu()

                for index in range(batch_size):
                    true_species_index = int(species_labels_cpu[index].item())
                    pred_species_index = int(species_preds_cpu[index].item())
                    true_domain_index = int(domain_labels_cpu[index].item())
                    pred_domain_index = int(domain_preds_cpu[index].item())
                    row = {
                        "file_id": batch["file_id"][index],
                        "true_species_index": true_species_index,
                        "predicted_species_index": pred_species_index,
                        "true_domain_index": true_domain_index,
                        "predicted_domain_index": pred_domain_index,
                        "species_logits": [float(value) for value in species_logits_cpu[index].tolist()],
                        "species_probabilities": [float(value) for value in species_probs_cpu[index].tolist()],
                    }
                    if species_names is not None:
                        row["true_species_label"] = species_names[true_species_index]
                        row["predicted_species_label"] = species_names[pred_species_index]
                    if domain_names is not None:
                        row["true_domain_label"] = domain_names[true_domain_index]
                        row["predicted_domain_label"] = domain_names[pred_domain_index]
                    prediction_rows.append(row)

    species_preds = torch.cat(species_preds)
    species_labels = torch.cat(species_labels)
    domain_preds = torch.cat(domain_preds)
    domain_labels = torch.cat(domain_labels)

    metrics = {
        "loss": total_loss / max(total_items, 1),
        "species_loss": total_species_loss / max(total_items, 1),
        "domain_loss": total_domain_loss / max(total_items, 1),
        "species_accuracy": accuracy(species_preds, species_labels),
        "species_balanced_accuracy": balanced_accuracy(species_preds, species_labels, num_species_classes),
        "domain_accuracy": accuracy(domain_preds, domain_labels),
        "domain_balanced_accuracy": balanced_accuracy(domain_preds, domain_labels, num_domain_classes),
    }

    species_domain_buckets = defaultdict(lambda: {"preds": [], "labels": []})
    for pred, label, domain_name in zip(species_preds.tolist(), species_labels.tolist(), batch_domain_names):
        species_domain_buckets[domain_name]["preds"].append(pred)
        species_domain_buckets[domain_name]["labels"].append(label)

    for domain_name, values in species_domain_buckets.items():
        domain_species_preds = torch.tensor(values["preds"])
        domain_species_labels = torch.tensor(values["labels"])
        metrics[f"species_ba_{domain_name}"] = balanced_accuracy(
            domain_species_preds,
            domain_species_labels,
            num_species_classes,
        )

    if return_predictions:
        metrics["num_prediction_rows"] = len(prediction_rows)
        return {
            "metrics": metrics,
            "predictions": prediction_rows,
        }
    return metrics


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    mixup_fn=None,
    fbs_mix_fn=None,
    grl_lambda=None,
    domain_loss_weight: float = 1.0,
    scol_weight: float = 0.0,
    scol_tau: float = 0.01,
    dicl_weight: float = 0.0,
    dicl_tau: float = 0.07,
    sdal_weight: float = 0.0,
    sdal_sigma: float = 1.0,
    wingbeat_weight: float = 0.0,
    group_dro_state: "GroupDROState | None" = None,
    species_loss_weight: "torch.Tensor | None" = None,
) -> dict:
    """Train for one epoch.

    Args:
        mixup_fn:   optional callable matching the signature of
            ``augmentation.mixup_batch``. When provided, Mixup is applied to
            every batch and accuracy is computed on the primary (un-permuted)
            labels only.
        grl_lambda: if not None, sets the GRL coefficient on the model via
            ``model.set_grl_lambda(grl_lambda)`` before the epoch starts.
            Pass the output of ``gradient_reversal.dann_lambda`` each epoch.
    """
    if grl_lambda is not None and hasattr(model, "set_grl_lambda"):
        model.set_grl_lambda(grl_lambda)
    model.train()
    total_loss = 0.0
    total_species_loss = 0.0
    total_domain_loss = 0.0
    total_species_correct = 0
    total_domain_correct = 0
    total_items = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        species_labels = batch["species_labels"].to(device)
        domain_labels = batch["domain_labels"].to(device)

        # FBS-Mix: frequency-band selective style mixing (input-level, before model)
        # lengths passed so statistics are computed on valid frames only (not padding)
        if fbs_mix_fn is not None:
            features = fbs_mix_fn(features, lengths)

        if mixup_fn is not None:
            features, sp_a, sp_b, dom_a, dom_b, lam = mixup_fn(features, species_labels, domain_labels)
        else:
            sp_a, sp_b, dom_a, dom_b, lam = species_labels, species_labels, domain_labels, domain_labels, 1.0

        optimizer.zero_grad()
        outputs = model(features, lengths)
        species_logits = outputs["species_logits"]
        domain_logits = outputs["domain_logits"]

        sw = species_loss_weight  # None or [num_species] weight tensor
        if group_dro_state is not None and mixup_fn is None:
            # GroupDRO: q-weighted per-domain species loss (no mixup support)
            species_loss = group_dro_state.weighted_loss(species_logits, sp_a, domain_labels)
        else:
            species_loss = (
                lam * F.cross_entropy(species_logits, sp_a, weight=sw)
                + (1.0 - lam) * F.cross_entropy(species_logits, sp_b, weight=sw)
            )
        domain_loss = (
            lam * F.cross_entropy(domain_logits, dom_a)
            + (1.0 - lam) * F.cross_entropy(domain_logits, dom_b)
        )
        loss = species_loss + domain_loss_weight * domain_loss

        # Contrastive / alignment losses operate on raw (non-mixed) labels.
        # Skip when mixup is active — mixed labels break contrastive pairing.
        if (scol_weight > 0 or dicl_weight > 0 or sdal_weight > 0) and mixup_fn is None:
            emb = outputs.get("proj_embedding", outputs.get("embedding"))
            if emb is not None:
                if scol_weight > 0:
                    loss = loss + scol_weight * species_cohesion_contrastive_loss(
                        emb, species_labels, tau=scol_tau
                    )
                if dicl_weight > 0:
                    loss = loss + dicl_weight * domain_invariant_contrastive_loss(
                        emb, species_labels, domain_labels, tau=dicl_tau
                    )
                if sdal_weight > 0:
                    loss = loss + sdal_weight * species_conditional_mmd_loss(
                        emb, species_labels, domain_labels, sigma=sdal_sigma
                    )

        if wingbeat_weight > 0 and mixup_fn is None:
            wb_pred = outputs.get("wingbeat_pred")
            if wb_pred is not None:
                wb_targets = SPECIES_WINGBEAT_TARGETS.to(device)[species_labels]  # [B]
                loss = loss + wingbeat_weight * F.mse_loss(wb_pred.squeeze(1), wb_targets)

        loss.backward()
        optimizer.step()

        species_preds = torch.argmax(species_logits, dim=1)
        domain_preds  = torch.argmax(domain_logits, dim=1)
        batch_size = sp_a.size(0)

        total_loss += loss.item() * batch_size
        total_species_loss += species_loss.item() * batch_size
        total_domain_loss += domain_loss.item() * batch_size
        total_species_correct += (species_preds == sp_a).sum().item()
        total_domain_correct  += (domain_preds  == dom_a).sum().item()
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "species_loss": total_species_loss / max(total_items, 1),
        "domain_loss": total_domain_loss / max(total_items, 1),
        "species_accuracy": total_species_correct / max(total_items, 1),
        "domain_accuracy": total_domain_correct / max(total_items, 1),
    }
