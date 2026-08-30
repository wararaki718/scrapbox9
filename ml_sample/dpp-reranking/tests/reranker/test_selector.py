import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.reranker import GreedyMapSelector


class GreedyMapSelectorTests(unittest.TestCase):
    def test_selects_quality_then_complementary_candidate(self) -> None:
        kernel = torch.tensor(
            [[4.0, 3.9, 0.0], [3.9, 4.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertEqual(GreedyMapSelector().select(kernel, 2), [0, 2])

    def test_ties_follow_original_candidate_order(self) -> None:
        self.assertEqual(GreedyMapSelector().select(torch.eye(3), 2), [0, 1])

    def test_rejects_invalid_selection_count(self) -> None:
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.eye(2), 3)

    def test_rejects_non_square_and_non_finite_kernels(self) -> None:
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.ones(2, 3), 1)
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.tensor([[float("nan")]]), 1)


if __name__ == "__main__":
    unittest.main()