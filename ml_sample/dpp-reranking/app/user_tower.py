import torch
from torch import nn
from torch.nn import functional


class UserTower(nn.Module):
    def __init__(self, num_users: int, embedding_dim: int) -> None:
        super().__init__()
        if num_users <= 0 or embedding_dim <= 0:
            raise ValueError("num_users and embedding_dim must be positive")
        self.embedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.embedding(user_ids), dim=-1)