from collections.abc import Mapping, Sequence

import torch
from torch.nn import functional


def intra_list_diversity_at_k(
    item_ids: Sequence[int],
    embedding_by_item_id: Mapping[int, torch.Tensor],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")

    selected_item_ids = item_ids[:k]
    if len(selected_item_ids) < 2:
        return 0.0
    try:
        embeddings = [embedding_by_item_id[item_id] for item_id in selected_item_ids]
    except KeyError as error:
        raise ValueError("all selected items must have embeddings") from error
    if any(embedding.ndim != 1 for embedding in embeddings):
        raise ValueError("embeddings must be vectors")
    dimensions = {embedding.shape[0] for embedding in embeddings}
    if len(dimensions) != 1:
        raise ValueError("embeddings must have the same dimension")
    stacked = torch.stack(embeddings)
    if not torch.isfinite(stacked).all():
        raise ValueError("embeddings must be finite")

    normalized = functional.normalize(stacked, dim=1)
    distances = 1.0 - normalized @ normalized.T
    row_indices, column_indices = torch.triu_indices(
        len(embeddings), len(embeddings), offset=1
    )
    return float(distances[row_indices, column_indices].mean().item())