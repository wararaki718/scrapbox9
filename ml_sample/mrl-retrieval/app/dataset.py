import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class PairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer: Tokenizer,
        max_length: int,
    ) -> None:
        if not pairs:
            raise ValueError("pairs must not be empty")
        self._pairs = pairs
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        query_ids, query_mask = self._tokenizer.encode(self._pairs[index][0], self._max_length)
        positive_ids, positive_mask = self._tokenizer.encode(self._pairs[index][1], self._max_length)
        return {
            "query_ids": query_ids,
            "query_mask": query_mask,
            "positive_ids": positive_ids,
            "positive_mask": positive_mask,
        }