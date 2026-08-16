import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from .model import MatryoshkaEncoder


def evaluate(
    model: MatryoshkaEncoder,
    loader: DataLoader[dict[str, torch.Tensor]],
    dimensions: list[int],
    device: torch.device,
) -> dict[int, float]:
    model.eval()
    query_embeddings, positive_embeddings = [], []
    with torch.no_grad():
        for batch in loader:
            query_embeddings.append(model(batch["query_ids"].to(device), batch["query_mask"].to(device)).cpu())
            positive_embeddings.append(model(batch["positive_ids"].to(device), batch["positive_mask"].to(device)).cpu())

    queries, positives = torch.cat(query_embeddings), torch.cat(positive_embeddings)
    results = {}
    for dimension in dimensions:
        query_prefix = functional.normalize(queries[:, :dimension], dim=1)
        positive_prefix = functional.normalize(positives[:, :dimension], dim=1)
        similarities = query_prefix @ positive_prefix.T
        correct = similarities.argmax(dim=1) == torch.arange(len(queries))
        results[dimension] = float(correct.float().mean())
        print(f"dimension={dimension} Recall@1={results[dimension]:.4f}")
    return results
