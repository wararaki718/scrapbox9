import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as functional


def matryoshka_cross_entropy(
    logits_by_dimension: Mapping[int, Tensor],
    labels: Tensor,
    relative_importance: Sequence[float],
) -> Tensor:
    if not logits_by_dimension:
        raise ValueError("logits_by_dimension must not be empty")
    if not relative_importance:
        raise ValueError("relative_importance must not be empty")
    if len(relative_importance) != len(logits_by_dimension):
        raise ValueError("relative_importance length must match logits_by_dimension")
    if any(not math.isfinite(weight) or weight < 0 for weight in relative_importance):
        raise ValueError("relative_importance values must be finite and nonnegative")
    if not isinstance(labels, Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a rank-one tensor")
    if labels.numel() == 0:
        raise ValueError("labels must not be empty")
    if labels.dtype != torch.long:
        raise ValueError("labels must have dtype torch.long")
    if torch.any(labels < 0) or torch.any(labels >= 10):
        raise ValueError("labels must be in the range [0, 10)")

    losses = []
    reference_logits = None
    for weight, logits in zip(relative_importance, logits_by_dimension.values(), strict=True):
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError("each logits tensor must be rank two")
        if logits.shape[1] != 10:
            raise ValueError("each logits tensor must have 10 columns")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("each logits tensor batch size must match labels")
        if logits.device != labels.device:
            raise ValueError("each logits tensor must share the labels device")
        if not logits.is_floating_point():
            raise ValueError("each logits tensor must have a floating dtype")

        if weight == 0:
            continue
        if reference_logits is None:
            reference_logits = logits
        elif logits.device != reference_logits.device or logits.dtype != reference_logits.dtype:
            raise ValueError("nonzero-weight logits tensors must share device and dtype")
        losses.append(weight * functional.cross_entropy(logits, labels))

    if not losses:
        return sum(
            torch.where(
                torch.zeros_like(logits, dtype=torch.bool), logits, torch.zeros_like(logits)
            ).sum()
            for logits in logits_by_dimension.values()
        )

    return torch.stack(losses).sum()