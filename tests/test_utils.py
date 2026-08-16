import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import CharacterTokenizer
from utils import PairDataset, evaluate_recall_at_one, load_pairs


class UtilsTests(unittest.TestCase):
    def test_jsonl_and_dataset(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
            file.write(json.dumps({"query": "質問", "positive": "回答"}) + "\n")
            file.flush()
            self.assertEqual(load_pairs(Path(file.name)), [("質問", "回答")])
        tokenizer = CharacterTokenizer().fit(["質問", "回答"])
        self.assertEqual(PairDataset([("質問", "回答")], tokenizer, 4)[0]["query_ids"].shape, (4,))

    def test_recall(self) -> None:
        self.assertEqual(evaluate_recall_at_one(torch.eye(2), torch.eye(2)), 1.0)
