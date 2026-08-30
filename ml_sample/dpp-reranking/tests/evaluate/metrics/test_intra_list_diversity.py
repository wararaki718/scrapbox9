import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.evaluate.metrics.intra_list_diversity import intra_list_diversity_at_k


class IntraListDiversityTests(unittest.TestCase):
    def test_returns_mean_pairwise_cosine_distance(self) -> None:
        embeddings = {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
        }

        self.assertAlmostEqual(intra_list_diversity_at_k([1, 2], embeddings, 2), 1.0)

    def test_returns_zero_for_single_item(self) -> None:
        self.assertEqual(
            intra_list_diversity_at_k([1], {1: torch.tensor([1.0])}, 1), 0.0
        )

    def test_rejects_invalid_embeddings(self) -> None:
        invalid_embeddings = (
            {1: torch.tensor([1.0])},
            {1: torch.tensor([1.0]), 2: torch.tensor([1.0, 0.0])},
            {1: torch.tensor([1.0]), 2: torch.tensor([float("nan")])},
        )
        for embeddings in invalid_embeddings:
            with self.subTest(embeddings=embeddings), self.assertRaises(ValueError):
                intra_list_diversity_at_k([1, 2], embeddings, 2)

    def test_rejects_non_positive_cutoff(self) -> None:
        with self.assertRaises(ValueError):
            intra_list_diversity_at_k([], {}, 0)


if __name__ == "__main__":
    unittest.main()