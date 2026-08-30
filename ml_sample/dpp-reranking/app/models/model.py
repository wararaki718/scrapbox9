import torch
from torch import nn

from app.models.item_tower import ItemTower
from app.models.user_tower import UserTower


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, embedding_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        users = self.user_tower(user_ids)
        items = self.item_tower(item_ids)
        return (users * items).sum(dim=-1)