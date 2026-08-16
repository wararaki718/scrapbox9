from collections.abc import Sequence

import torch
from torch.nn import functional as functional


def matryoshka_infonce_loss(
    queries: torch.Tensor,
    positives: torch.Tensor,
    dimensions: Sequence[int],
    temperature: float,
) -> torch.Tensor:
    if queries.ndim != 2 or queries.shape != positives.shape or not dimensions or temperature <= 0:
        raise ValueError("invalid embeddings, dimensions, or temperature")

    labels = torch.arange(queries.shape[0], device=queries.device)
    losses = []
    for dimension in dimensions:
        if not 0 < dimension <= queries.shape[1]:
            raise ValueError("invalid embedding dimension")
        logits = functional.normalize(queries[:, :dimension], dim=1) @ functional.normalize(positives[:, :dimension], dim=1).T / temperature
        losses.append((functional.cross_entropy(logits, labels) + functional.cross_entropy(logits.T, labels)) / 2)
    return torch.stack(losses).mean()