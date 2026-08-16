import torch

from loss import matryoshka_infonce_loss
from model import MatryoshkaEncoder


class Trainer:
    def __init__(self, model: MatryoshkaEncoder, dimensions: list[int], temperature: float, learning_rate: float, device: torch.device) -> None:
        self.model, self.dimensions, self.temperature, self.device = model.to(device), dimensions, temperature, device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    def train_epoch(self, loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]]) -> float:
        total_loss = 0.0
        self.model.train()
        for count, batch in enumerate(loader, 1):
            batch = {key: value.to(self.device) for key, value in batch.items()}
            queries = self.model(batch["query_ids"], batch["query_mask"])
            positives = self.model(batch["positive_ids"], batch["positive_mask"])
            loss = matryoshka_infonce_loss(queries, positives, self.dimensions, self.temperature)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        if not total_loss and count == 0:
            raise ValueError("loader must not be empty")
        return total_loss / count