import json
import random
from pathlib import Path

import torch

DEFAULT_PAIRS = [("東京の天気", "東京では今日、晴れの予報です。"), ("Pythonでリストを並べ替える", "Pythonのlist.sortメソッドはリストをその場で並べ替えます。"), ("富士山の高さ", "富士山の標高はおよそ3776メートルです。"), ("機械学習とは", "機械学習はデータから規則を学ぶ人工知能の手法です。")]


def load_pairs(path: Path | None) -> list[tuple[str, str]]:
    if path is None:
        return DEFAULT_PAIRS

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

