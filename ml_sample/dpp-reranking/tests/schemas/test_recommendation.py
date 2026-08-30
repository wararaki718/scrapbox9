import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.schemas import Recommendation


class RecommendationTests(unittest.TestCase):
    def test_is_immutable(self) -> None:
        recommendation = Recommendation(1, "books", 0.5)

        with self.assertRaises(AttributeError):
            recommendation.score = 1.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()