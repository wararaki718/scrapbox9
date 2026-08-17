from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as functional


def matryoshka_cross_entropy(logits_by_dimension: Mapping[int, Tensor], labels: Tensor) -> Tensor:
    if not logits_by_dimension:
        raise ValueError("logits_by_dimension must not be empty")
    if not isinstance(labels, Tensor) or labels.ndim != 1:
        raise ValueError("labels must be a rank-one tensor")
    if labels.numel() == 0:
        raise ValueError("labels must not be empty")
    if labels.dtype != torch.long:
        raise ValueError("labels must have dtype torch.long")
    if torch.any(labels < 0) or torch.any(labels >= 10):
        raise ValueError("labels must be in the range [0, 10)")

    losses = []
    for logits in logits_by_dimension.values():
        if not isinstance(logits, Tensor) or logits.ndim != 2:
            raise ValueError("each logits tensor must be rank two")
        if logits.shape[1] != 10:
            raise ValueError("each logits tensor must have 10 columns")
        if logits.shape[0] != labels.shape[0]:
            raise ValueError("each logits tensor batch size must match labels")
        losses.append(functional.cross_entropy(logits, labels))

    return torch.stack(losses).mean()