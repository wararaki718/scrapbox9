import torch
from torch import nn
from torch.nn import functional


class ItemTower(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        if num_items <= 0 or embedding_dim <= 0:
            raise ValueError("num_items and embedding_dim must be positive")
        self.embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.embedding(item_ids), dim=-1)