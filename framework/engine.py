"""Training and evaluation utilities for the DCASE2026 mosquito baseline.

Author: Yuanbo Hou
Email: Yuanbo.Hou@eng.ox.ac.uk
Affiliation: Machine Learning Research Group, University of Oxford
"""

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


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
) -> dict:
    model.eval()
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

            outputs = model(features, lengths, species_labels=batch_species_labels)
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


def dann_alpha(epoch: int, total_epochs: int, alpha_max: float) -> float:
    """Ganin et al. 2016 lambda schedule: ramps from 0 to alpha_max over training."""
    p = epoch / total_epochs
    return alpha_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def supcon_loss(projections: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al. 2020).

    Uses all same-class samples in the batch as positives — no augmented-view pairs needed.
    Anchors with no in-batch positive are excluded from the loss.
    projections: [B, D] L2-normalised embeddings from the projection head
    labels: [B] integer species labels
    """
    B = projections.shape[0]
    sim = torch.mm(projections, projections.T) / temperature
    self_mask = torch.eye(B, dtype=torch.bool, device=projections.device)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~self_mask
    log_denom = torch.logsumexp(sim.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True)
    log_prob = sim - log_denom
    num_pos = pos_mask.float().sum(dim=1)
    has_pos = num_pos > 0
    if not has_pos.any():
        return projections.sum() * 0.0
    per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / num_pos.clamp(min=1)
    return per_anchor[has_pos].mean()


def train_one_epoch(model, dataloader, optimizer, device, epoch: int = 1, total_epochs: int = 100, dann_alpha_max: float = 0.0, supcon_weight: float = 0.0, supcon_temperature: float = 0.1) -> dict:
    alpha = dann_alpha(epoch, total_epochs, dann_alpha_max) if dann_alpha_max > 0.0 else None
    model.train()
    total_loss = 0.0
    total_species_loss = 0.0
    total_domain_loss = 0.0
    total_supcon_loss = 0.0
    total_species_correct = 0
    total_domain_correct = 0
    total_items = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        species_labels = batch["species_labels"].to(device)
        domain_labels = batch["domain_labels"].to(device)

        optimizer.zero_grad()
        outputs = model(features, lengths, alpha=alpha, species_labels=species_labels)
        species_logits = outputs["species_logits"]
        domain_logits = outputs["domain_logits"]

        species_loss = F.cross_entropy(species_logits, species_labels)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        loss = species_loss + domain_loss
        sc_loss_val = 0.0
        if supcon_weight > 0.0 and "projection" in outputs:
            sc = supcon_loss(outputs["projection"], species_labels, supcon_temperature)
            loss = loss + supcon_weight * sc
            sc_loss_val = sc.item()
        loss.backward()
        optimizer.step()

        species_preds = torch.argmax(species_logits, dim=1)
        domain_preds = torch.argmax(domain_logits, dim=1)
        batch_size = species_labels.size(0)

        total_loss += loss.item() * batch_size
        total_species_loss += species_loss.item() * batch_size
        total_domain_loss += domain_loss.item() * batch_size
        total_supcon_loss += sc_loss_val * batch_size
        total_species_correct += (species_preds == species_labels).sum().item()
        total_domain_correct += (domain_preds == domain_labels).sum().item()
        total_items += batch_size

    return {
        "loss": total_loss / max(total_items, 1),
        "species_loss": total_species_loss / max(total_items, 1),
        "domain_loss": total_domain_loss / max(total_items, 1),
        "supcon_loss": total_supcon_loss / max(total_items, 1),
        "species_accuracy": total_species_correct / max(total_items, 1),
        "domain_accuracy": total_domain_correct / max(total_items, 1),
    }
