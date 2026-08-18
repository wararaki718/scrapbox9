import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import MatryoshkaImageClassifier
from app.train import Trainer


class FixedPrefixLogitModel(torch.nn.Module):
    dimensions = [4, 8]

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "prefix_four_logits",
            torch.tensor(
                [
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [10.0, 0.0, 0.0],
                ]
            ),
        )
        self.register_buffer(
            "prefix_eight_logits",
            torch.tensor(
                [
                    [0.0, 10.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [10.0, 0.0, 0.0],
                ]
            ),
        )

    def forward(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        sample_ids = images[:, 0, 0, 0].long()
        return {
            4: self.prefix_four_logits[sample_ids] + self.anchor * 0,
            8: self.prefix_eight_logits[sample_ids] + self.anchor * 0,
        }


class TrainerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MatryoshkaImageClassifier(embedding_dim=8, dimensions=[4, 8])
        self.trainer = Trainer(self.model, torch.device("cpu"), 0.01, [1.0, 1.0])

    def test_train_epoch_updates_projection_and_returns_finite_loss(self) -> None:
        loader = DataLoader(
            TensorDataset(torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])),
            batch_size=2,
        )
        initial_weight = self.model.projection.weight.detach().clone()

        loss = self.trainer.train_epoch(loader)

        self.assertFalse(torch.equal(self.model.projection.weight, initial_weight))
        self.assertTrue(math.isfinite(loss))

    def test_evaluate_aggregates_correct_predictions_per_prefix(self) -> None:
        images = torch.zeros(3, 3, 32, 32)
        images[:, 0, 0, 0] = torch.tensor([0.0, 1.0, 2.0])
        loader = DataLoader(
            TensorDataset(images, torch.tensor([0, 1, 2])),
            batch_size=2,
        )
        trainer = Trainer(FixedPrefixLogitModel(), torch.device("cpu"), 0.01, [1.0, 1.0])

        accuracies = trainer.evaluate(loader)

        self.assertAlmostEqual(accuracies[4], 2 / 3)
        self.assertAlmostEqual(accuracies[8], 1 / 3)

    def test_empty_loader_raises_value_error(self) -> None:
        loader = DataLoader(
            TensorDataset(torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)),
            batch_size=2,
        )

        with self.assertRaises(ValueError):
            self.trainer.train_epoch(loader)
        with self.assertRaises(ValueError):
            self.trainer.evaluate(loader)

    def test_constructor_rejects_non_positive_or_non_finite_learning_rate(self) -> None:
        for learning_rate in (0.0, -0.01, float("inf"), float("nan")):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaises(ValueError):
                    Trainer(self.model, torch.device("cpu"), learning_rate, [1.0, 1.0])

    def test_constructor_rejects_invalid_relative_importance(self) -> None:
        for relative_importance in ([1.0], [1.0, 1.0, 1.0], [1.0, -1.0], [1.0, float("inf")]):
            with self.subTest(relative_importance=relative_importance):
                with self.assertRaises(ValueError):
                    Trainer(self.model, torch.device("cpu"), 0.01, relative_importance)

    def test_train_epoch_passes_relative_importance_to_loss(self) -> None:
        loader = DataLoader(
            TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1])),
            batch_size=2,
        )
        trainer = Trainer(self.model, torch.device("cpu"), 0.01, [2.0, 1.0])

        with patch(
            "app.train.matryoshka_cross_entropy",
            wraps=__import__("app.loss", fromlist=["matryoshka_cross_entropy"]).matryoshka_cross_entropy,
        ) as cross_entropy:
            trainer.train_epoch(loader)

        self.assertEqual(cross_entropy.call_args.args[2], (2.0, 1.0))


if __name__ == "__main__":
    unittest.main()