"""Test-Time Entropy Minimization (TENT) for domain adaptation at inference.

TENT (Wang et al., 2021 — https://arxiv.org/abs/2006.10726) adapts a model to
a new test distribution by minimizing the entropy of softmax predictions over
unlabelled test data. Only the affine parameters of batch/layer normalisation
layers are updated — all other weights stay frozen.

In this codebase TENT is applied to the unlabelled evaluation clips before
producing final predictions for submission. This is legally valid for the
challenge since it uses only the provided (unlabelled) evaluation audio.

Research motivation
-------------------
Training BN running statistics are computed on D5-dominated data (99.4%).
At test time, clips from D1–D4 are poorly served by these D5-biased stats.
TENT corrects the affine shift by minimising prediction entropy on the
actual test distribution, effectively performing unsupervised domain
adaptation without any labels or retraining.

Usage
-----
    from framework.tent import configure_tent, tent_step

    model = configure_tent(model)
    for batch in eval_loader:
        tent_step(model, batch["features"], batch["lengths"], optimizer, n_steps=1)
    # Now run normal inference — BN affine params are adapted.
"""

from typing import List

import torch
import torch.nn as nn
import torch.optim as optim


def _get_norm_layers(model: nn.Module) -> List[nn.Module]:
    """Return all BatchNorm and LayerNorm layers in the model."""
    return [
        m for m in model.modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm))
    ]


def configure_tent(model: nn.Module, lr: float = 1e-3) -> optim.Optimizer:
    """Prepare model for TENT: freeze all params, enable affine params only.

    Args:
        model: trained model (weights frozen after this call except BN/LN affine).
        lr:    learning rate for entropy minimisation steps.

    Returns:
        Adam optimiser over the unfrozen affine params.
    """
    model.train()

    # Freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Unfreeze only BN/LN affine (weight = γ, bias = β)
    norm_layers = _get_norm_layers(model)
    if not norm_layers:
        raise RuntimeError(
            "No BatchNorm or LayerNorm layers found — TENT has nothing to adapt."
        )

    for layer in norm_layers:
        if hasattr(layer, "weight") and layer.weight is not None:
            layer.weight.requires_grad_(True)
        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.requires_grad_(True)
        # Use batch statistics (not running stats) during TENT
        if hasattr(layer, "track_running_stats"):
            layer.track_running_stats = False

    n_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"TENT: {len(norm_layers)} norm layers, {n_params:,} params unfrozen "
        f"(all others frozen)"
    )

    return optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )


@torch.enable_grad()
def tent_step(
    model: nn.Module,
    features: torch.Tensor,
    lengths: torch.Tensor,
    optimizer: optim.Optimizer,
    n_steps: int = 1,
) -> float:
    """Run n_steps of entropy minimisation on a batch of unlabelled clips.

    Args:
        model:     model in train() mode with only norm affine params unfrozen.
        features:  padded log-mel spectrogram [B, T, F].
        lengths:   true frame counts [B].
        optimizer: Adam over norm affine params.
        n_steps:   gradient steps to take on this batch.

    Returns:
        Mean entropy value (for logging).
    """
    total_entropy = 0.0
    for _ in range(n_steps):
        optimizer.zero_grad()
        out = model(features, lengths)
        probs = out["species_logits"].softmax(dim=-1)      # [B, C]
        # Shannon entropy H = -Σ p log p, mean over batch
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
        entropy.backward()
        optimizer.step()
        total_entropy += entropy.item()
    return total_entropy / n_steps
