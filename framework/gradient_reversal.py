"""Gradient Reversal Layer (GRL) for domain-adversarial training (DANN).

Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks", JMLR 2016.

The GRL acts as an identity in the forward pass and multiplies gradients by -λ
in the backward pass.  Inserting it between the shared feature extractor and the
domain classifier forces the feature extractor to learn domain-invariant
representations.

λ is typically annealed from 0 → λ_max using the schedule from the DANN paper::

    λ(p) = λ_max × (2 / (1 + exp(-10 × p)) - 1),   p = epoch / total_epochs

This delays adversarial pressure until the feature extractor has learned basic
species discriminability, which stabilises early training.
"""

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------

class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


# ---------------------------------------------------------------------------
# Module wrapper
# ---------------------------------------------------------------------------

class GradientReversalLayer(nn.Module):
    """Identity forward, gradient-reversing backward.

    Set ``lambda_`` to 0.0 to disable (pure identity).
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_: float) -> None:
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)

    def extra_repr(self) -> str:
        return f"lambda={self.lambda_:.4f}"


# ---------------------------------------------------------------------------
# λ schedule
# ---------------------------------------------------------------------------

def dann_lambda(epoch: int, total_epochs: int, lambda_max: float = 1.0, gamma: float = 10.0) -> float:
    """DANN annealing schedule for the GRL coefficient.

    Args:
        epoch:        current epoch (1-indexed)
        total_epochs: total number of training epochs
        lambda_max:   maximum value of λ (scales the reversal strength)
        gamma:        steepness of the sigmoid ramp (default 10 from DANN paper)

    Returns:
        λ ∈ [0, lambda_max)
    """
    p = epoch / total_epochs
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)
