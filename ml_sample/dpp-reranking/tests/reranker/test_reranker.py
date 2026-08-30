import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.reranker import DPPReranker


class DPPRerankerTests(unittest.TestCase):
    def test_reranks_candidates_through_builder_and_selector(self) -> None:
        indices = DPPReranker().rerank(
            torch.tensor([1.0, 0.9, 0.1]),
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            top_k=2,
        )

        self.assertEqual(indices, [0, 2])


if __name__ == "__main__":
    unittest.main()