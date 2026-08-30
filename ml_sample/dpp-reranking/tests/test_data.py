import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import TrainConfig
from app.data import create_sample_data
from app.schemas import Recommendation


class DataTests(unittest.TestCase):
    def test_sample_data_has_contiguous_ids_and_valid_interactions(self) -> None:
        data = create_sample_data()

        self.assertEqual([item.item_id for item in data.items], list(range(len(data.items))))
        self.assertGreaterEqual(len({item.category for item in data.items}), 3)
        self.assertTrue(all(0 <= row.user_id < data.num_users for row in data.interactions))
        self.assertTrue(all(0 <= row.item_id < len(data.items) for row in data.interactions))

    def test_recommendation_is_immutable(self) -> None:
        recommendation = Recommendation(1, "books", 0.5)
        with self.assertRaises(AttributeError):
            recommendation.score = 1.0  # type: ignore[misc]

    def test_train_config_rejects_invalid_values(self) -> None:
        for kwargs in (
            {"embedding_dim": 0},
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"learning_rate": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()