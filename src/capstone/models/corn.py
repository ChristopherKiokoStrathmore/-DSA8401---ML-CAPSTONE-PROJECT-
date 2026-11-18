"""Conditional ordinal regression (CORN) for the three urgency levels.

Urgency is ordered: low < medium < emergency. CORN (Cao, Mirjalili, Raschka)
trains K-1 binary tasks whose targets are the conditional events
P(y > k | y > k-1). Class probabilities come from that chain. They are what
the emergency threshold reads.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

URGENCY_LEVELS: tuple[str, ...] = ("low", "medium", "emergency")


def corn_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int = 3,
    sample_weight: torch.Tensor | None = None,
    class_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean weighted CORN loss.

    ``labels`` are integers in ``0 .. num_classes-1``. ``logits`` has shape
    ``(n, num_classes - 1)``. Rows with a non-positive sample weight are skipped.
    """
    if logits.ndim != 2 or logits.size(1) != num_classes - 1:
        raise ValueError("CORN logits must have shape (n, num_classes - 1)")
    total = logits.new_zeros(())
    weight_sum = logits.new_zeros(())
    for rank in range(num_classes - 1):
        mask = labels > rank - 1
        if sample_weight is not None:
            mask = mask & (sample_weight > 0)
        if not bool(mask.any()):
            continue
        target = (labels[mask] > rank).to(logits.dtype)
        per_example = F.binary_cross_entropy_with_logits(
            logits[mask, rank], target, reduction="none"
        )
        weights = torch.ones_like(per_example)
        if sample_weight is not None:
            weights = weights * sample_weight[mask].to(logits.dtype)
        if class_weight is not None:
            weights = weights * class_weight[labels[mask]].to(logits.dtype)
        total = total + (per_example * weights).sum()
        weight_sum = weight_sum + weights.sum()
    if float(weight_sum) == 0.0:
        return logits.sum() * 0.0
    return total / weight_sum


def corn_class_probs(logits: torch.Tensor) -> torch.Tensor:
    """Return class probabilities with shape ``(n, K)`` from CORN logits.

    P(y > k) is the product of the conditional sigmoids up to task k.
    P(y = 0) = 1 - P(y > 0), and P(y = k) = P(y > k-1) - P(y > k).
    """
    conditional = torch.sigmoid(logits)
    exceed = torch.cumprod(conditional, dim=1)
    ones = torch.ones(logits.size(0), 1, device=logits.device, dtype=logits.dtype)
    zeros = torch.zeros(logits.size(0), 1, device=logits.device, dtype=logits.dtype)
    survival = torch.cat([ones, exceed, zeros], dim=1)
    return (survival[:, :-1] - survival[:, 1:]).clamp_min(0.0)


def corn_predict(logits: torch.Tensor) -> torch.Tensor:
    """Ordinal label, the class with the highest CORN probability."""
    return corn_class_probs(logits).argmax(dim=1)
