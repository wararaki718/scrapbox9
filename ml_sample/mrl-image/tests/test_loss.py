import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.loss import matryoshka_cross_entropy
from app.model import MatryoshkaImageClassifier


class MatryoshkaCrossEntropyTests(unittest.TestCase):
    def test_loss_matches_mean_of_cross_entropy_per_prefix(self) -> None:
        logits_by_dimension = {
            4: torch.tensor([[2.0, 0.5, -1.0, 0.0, 0.2, 0.3, 0.4, 0.1, -0.5, 1.0],
                             [0.0, 1.0, 2.0, -1.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.6]]),
            8: torch.tensor([[0.1, 1.5, 0.2, -0.5, 0.8, 0.0, 0.4, -1.0, 0.3, 0.6],
                             [1.2, -0.2, 0.4, 0.8, -0.1, 0.6, 0.0, 1.1, 0.3, -0.4]]),
        }
        labels = torch.tensor([1, 7])

        expected = torch.stack(
            [F.cross_entropy(logits, labels) for logits in logits_by_dimension.values()]
        ).mean()

        self.assertTrue(torch.allclose(matryoshka_cross_entropy(logits_by_dimension, labels), expected))

    def test_loss_backpropagates_to_every_classifier_head(self) -> None:
        model = MatryoshkaImageClassifier(embedding_dim=16, dimensions=[4, 8, 16])
        logits_by_dimension = model(torch.rand(3, 3, 32, 32))

        loss = matryoshka_cross_entropy(logits_by_dimension, torch.tensor([0, 1, 2]))
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for classifier in model.classifiers.values():
            self.assertIsNotNone(classifier.weight.grad)

    def test_loss_rejects_invalid_logit_shapes(self) -> None:
        labels = torch.tensor([0, 1, 2])
        for logits_by_dimension in (
            {4: torch.rand(2, 10)},
            {4: torch.rand(3, 9)},
        ):
            with self.subTest(logits_by_dimension=logits_by_dimension):
                with self.assertRaises(ValueError):
                    matryoshka_cross_entropy(logits_by_dimension, labels)

    def test_loss_rejects_invalid_labels_with_value_error(self) -> None:
        logits_by_dimension = {4: torch.rand(2, 10)}
        for labels in (
            torch.tensor([[0, 1]]),
            torch.tensor([], dtype=torch.long),
            torch.tensor([0.0, 1.0]),
            torch.tensor([-1, 1]),
            torch.tensor([0, 10]),
        ):
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    matryoshka_cross_entropy(logits_by_dimension, labels)


if __name__ == "__main__":
    unittest.main()