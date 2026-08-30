import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from app.evaluate.metrics.category_coverage import category_coverage_at_k


class CategoryCoverageTests(unittest.TestCase):
    def test_counts_distinct_categories_within_cutoff(self) -> None:
        categories = {1: "books", 2: "music", 3: "sports"}

        self.assertEqual(category_coverage_at_k([1, 2, 3], categories, 2), 2)

    def test_rejects_missing_category(self) -> None:
        with self.assertRaises(ValueError):
            category_coverage_at_k([1, 2], {1: "books"}, 2)

    def test_rejects_non_positive_cutoff(self) -> None:
        with self.assertRaises(ValueError):
            category_coverage_at_k([], {}, 0)


if __name__ == "__main__":
    unittest.main()