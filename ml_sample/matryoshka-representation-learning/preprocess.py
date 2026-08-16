from __future__ import annotations

import torch


class CharacterTokenizer:
    def __init__(self) -> None:
        self._vocabulary = {"<pad>": 0, "<unk>": 1}

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    @property
    def unknown_id(self) -> int:
        return self._vocabulary["<unk>"]

    def fit(self, texts: list[str]) -> CharacterTokenizer:
        for text in texts:
            for character in text:
                if character not in self._vocabulary:
                    self._vocabulary[character] = len(self._vocabulary)
        return self

    def encode(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        token_ids = [self._vocabulary.get(character, self.unknown_id) for character in text[:max_length]]
        attention_mask = [1] * len(token_ids)
        token_ids.extend([0] * (max_length - len(token_ids)))
        attention_mask.extend([0] * (max_length - len(attention_mask)))
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)