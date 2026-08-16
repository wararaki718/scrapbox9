import torch

from .loss import matryoshka_infonce_loss
from .model import MatryoshkaEncoder


class Trainer:
    def __init__(
        self, model: MatryoshkaEncoder,
        dimensions: list[int],
        temperature: float,
        learning_rate: float,
        device: torch.device,
    ) -> None:
        self._model = model.to(device)
        self._dimensions = dimensions
        self._temperature = temperature
        self._device = device
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)

    def train_epoch(self, loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]]) -> float:
        total_loss = 0.0
        self._model.train()
        for count, batch in enumerate(loader, 1):
            batch = {key: value.to(self._device) for key, value in batch.items()}
            queries = self._model(batch["query_ids"], batch["query_mask"])
            positives = self._model(batch["positive_ids"], batch["positive_mask"])
            loss = matryoshka_infonce_loss(queries, positives, self._dimensions, self._temperature)
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            total_loss += loss.item()
        if not total_loss and count == 0:
            raise ValueError("loader must not be empty")
        return total_loss / count
