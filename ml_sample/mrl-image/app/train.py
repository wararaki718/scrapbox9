import math
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader

from app.loss import matryoshka_cross_entropy
from app.model import MatryoshkaImageClassifier


class Trainer:
    def __init__(
        self,
        model: MatryoshkaImageClassifier,
        device: torch.device,
        learning_rate: float,
        relative_importance: Sequence[float],
    ) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if len(relative_importance) != len(model.dimensions):
            raise ValueError("relative_importance length must match model dimensions")
        if any(not math.isfinite(weight) or weight < 0 for weight in relative_importance):
            raise ValueError("relative_importance values must be finite and nonnegative")

        self.model = model.to(device)
        self.device = device
        self.relative_importance = tuple(relative_importance)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            logits_by_dimension = self.model(images)
            loss = matryoshka_cross_entropy(logits_by_dimension, labels, self.relative_importance)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        if batch_count == 0:
            raise ValueError("loader must not be empty")

        return total_loss / batch_count

    def evaluate(self, loader: DataLoader) -> dict[int, float]:
        self.model.eval()
        correct_by_dimension = {dimension: 0 for dimension in self.model.dimensions}
        total_examples = 0

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits_by_dimension = self.model(images)

                for dimension, logits in logits_by_dimension.items():
                    correct_by_dimension[dimension] += (logits.argmax(dim=1) == labels).sum().item()
                total_examples += labels.size(0)

        if total_examples == 0:
            raise ValueError("loader must not be empty")

        return {
            dimension: correct / total_examples
            for dimension, correct in correct_by_dimension.items()
        }