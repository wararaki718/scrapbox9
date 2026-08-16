import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset import PairDataset
from app.tokenizer import Tokenizer


class PairDatasetTests(unittest.TestCase):
    def test_returns_encoded_query_and_positive(self) -> None:
        tokenizer = Tokenizer().fit(["質問", "回答"])
        sample = PairDataset([("質問", "回答")], tokenizer, 4)[0]

        self.assertEqual(set(sample), {"query_ids", "query_mask", "positive_ids", "positive_mask"})
        self.assertEqual(sample["query_ids"].shape, (4,))


if __name__ == "__main__":
    unittest.main()