from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader, Dataset

from preprocess import CharacterTokenizer

DEFAULT_PAIRS = [("東京の天気", "東京では今日、晴れの予報です。"), ("Pythonでリストを並べ替える", "Pythonのlist.sortメソッドはリストをその場で並べ替えます。"), ("富士山の高さ", "富士山の標高はおよそ3776メートルです。"), ("機械学習とは", "機械学習はデータから規則を学ぶ人工知能の手法です。")]


def load_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
        if not isinstance(record, dict) or not isinstance(record.get("query"), str) or not isinstance(record.get("positive"), str):
            raise ValueError(f"line {line_number} must contain string query and positive fields")
        pairs.append((record["query"], record["positive"]))
    if not pairs:
        raise ValueError("dataset is empty")
    return pairs


class PairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer: CharacterTokenizer, max_length: int) -> None:
        if not pairs:
            raise ValueError("pairs must not be empty")
        self._pairs, self._tokenizer, self._max_length = pairs, tokenizer, max_length

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        query_ids, query_mask = self._tokenizer.encode(self._pairs[index][0], self._max_length)
        positive_ids, positive_mask = self._tokenizer.encode(self._pairs[index][1], self._max_length)
        return {"query_ids": query_ids, "query_mask": query_mask, "positive_ids": positive_ids, "positive_mask": positive_mask}


def create_dataloader(dataset: PairDataset, batch_size: int, shuffle: bool) -> DataLoader[dict[str, torch.Tensor]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate_recall_at_one(query_embeddings: torch.Tensor, positive_embeddings: torch.Tensor) -> float:
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError("embedding shapes must match")
    similarities = functional.normalize(query_embeddings, dim=1) @ functional.normalize(positive_embeddings, dim=1).T
    return float((similarities.argmax(dim=1) == torch.arange(len(query_embeddings), device=similarities.device)).float().mean())