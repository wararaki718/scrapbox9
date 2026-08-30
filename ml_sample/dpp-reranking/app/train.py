import random
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch.nn import functional

from app.config import TrainConfig
from app.model import TwoTowerModel
from app.schemas import Interaction


def train(
    model: TwoTowerModel,
    interactions: Sequence[Interaction],
    num_items: int,
    config: TrainConfig,
) -> list[float]:
    if not interactions:
        raise ValueError("interactions must not be empty")

    positives: dict[int, set[int]] = defaultdict(set)
    for interaction in interactions:
        positives[interaction.user_id].add(interaction.item_id)
    negatives = {
        user_id: tuple(item_id for item_id in range(num_items) if item_id not in item_ids)
        for user_id, item_ids in positives.items()
    }
    if any(not item_ids for item_ids in negatives.values()):
        raise ValueError("each user must have at least one negative item")

    torch.manual_seed(config.seed)
    random_generator = random.Random(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    epoch_losses: list[float] = []
    rows = list(interactions)

    for _ in range(config.epochs):
        random_generator.shuffle(rows)
        total_loss = 0.0
        for row in rows:
            negative_item_id = random_generator.choice(negatives[row.user_id])
            user_ids = torch.tensor([row.user_id])
            positive_ids = torch.tensor([row.item_id])
            negative_ids = torch.tensor([negative_item_id])
            loss = -functional.logsigmoid(
                model(user_ids, positive_ids) - model(user_ids, negative_ids)
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(rows))

    return epoch_losses