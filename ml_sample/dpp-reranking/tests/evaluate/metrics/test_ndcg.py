import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.evaluate.metrics.ndcg import ndcg_at_k


class NDCGTests(unittest.TestCase):
    def test_returns_one_for_ideal_ranking(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], {1: 3, 2: 2, 3: 1}, 3), 1.0)

    def test_penalizes_reversed_ranking(self) -> None:
        score = ndcg_at_k([3, 2, 1], {1: 3, 2: 2, 3: 1}, 3)

        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_treats_missing_judgments_as_zero(self) -> None:
        self.assertAlmostEqual(ndcg_at_k([1, 2], {1: 1}, 2), 1.0)

    def test_returns_zero_when_ideal_dcg_is_zero(self) -> None:
        self.assertEqual(ndcg_at_k([1, 2], {}, 2), 0.0)

    def test_rejects_non_positive_cutoff(self) -> None:
        with self.assertRaises(ValueError):
            ndcg_at_k([1], {1: 1}, 0)


if __name__ == "__main__":
    unittest.main()