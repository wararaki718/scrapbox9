from __future__ import annotations

import torch


class Tokenizer:
    def __init__(self) -> None:
        self._vocabulary = {"<pad>": 0, "<unk>": 1}

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    @property
    def unknown_id(self) -> int:
        return self._vocabulary["<unk>"]

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def fit(self, texts: list[str]) -> Tokenizer:
        for text in texts:
            for token in self.tokenize(text):
                if token not in self._vocabulary:
                    self._vocabulary[token] = len(self._vocabulary)
        return self

    def encode(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        tokens = self.tokenize(text)[:max_length]
        token_ids = [self._vocabulary.get(token, self.unknown_id) for token in tokens]
        attention_mask = [1] * len(token_ids)
        token_ids.extend([0] * (max_length - len(token_ids)))
        attention_mask.extend([0] * (max_length - len(attention_mask)))
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)