import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import TrainConfig


class TrainConfigTests(unittest.TestCase):
    def test_rejects_invalid_values(self) -> None:
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