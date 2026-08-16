import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluate import evaluate


class EvaluateTests(unittest.TestCase):
    def test_evaluate_returns_recall_for_every_dimension(self) -> None:
        class IdentityModel:
            def eval(self) -> None:
                pass

            def __call__(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                return token_ids.float()

        batch = {
            "query_ids": torch.tensor([[1, -1], [-1, 1]]),
            "query_mask": torch.ones((2, 2), dtype=torch.long),
            "positive_ids": torch.tensor([[1, -1], [-1, 1]]),
            "positive_mask": torch.ones((2, 2), dtype=torch.long),
        }
        self.assertEqual(evaluate(IdentityModel(), [batch], [1, 2], torch.device("cpu")), {1: 1.0, 2: 1.0})


if __name__ == "__main__":
    unittest.main()