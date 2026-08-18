import sys
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.loss import matryoshka_cross_entropy
from app.model import MatryoshkaImageClassifier


class MatryoshkaCrossEntropyTests(unittest.TestCase):
    def test_loss_matches_weighted_sum_of_cross_entropy_per_prefix(self) -> None:
        logits_by_dimension = {
            4: torch.tensor([[2.0, 0.5, -1.0, 0.0, 0.2, 0.3, 0.4, 0.1, -0.5, 1.0],
                             [0.0, 1.0, 2.0, -1.0, 0.5, 0.1, 0.2, 0.3, 0.4, 0.6]]),
            8: torch.tensor([[0.1, 1.5, 0.2, -0.5, 0.8, 0.0, 0.4, -1.0, 0.3, 0.6],
                             [1.2, -0.2, 0.4, 0.8, -0.1, 0.6, 0.0, 1.1, 0.3, -0.4]]),
        }
        labels = torch.tensor([1, 7])
        relative_importance = [2.0, 0.5]

        expected = torch.stack(
            [
                weight * F.cross_entropy(logits, labels)
                for weight, logits in zip(relative_importance, logits_by_dimension.values(), strict=True)
            ]
        ).sum()

        self.assertTrue(
            torch.allclose(
                matryoshka_cross_entropy(logits_by_dimension, labels, relative_importance), expected
            )
        )

    def test_loss_backpropagates_to_every_classifier_head(self) -> None:
        model = MatryoshkaImageClassifier(embedding_dim=16, dimensions=[4, 8, 16])
        logits_by_dimension = model(torch.rand(3, 3, 32, 32))

        loss = matryoshka_cross_entropy(logits_by_dimension, torch.tensor([0, 1, 2]), [1.0, 1.0, 1.0])
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        for classifier in model.classifiers.values():
            self.assertIsNotNone(classifier.weight.grad)

    def test_loss_skips_zero_weight_nan_logits_without_gradient(self) -> None:
        labels = torch.tensor([0, 1])
        zero_weight_logits = torch.full((2, 10), float("nan"), requires_grad=True)
        weighted_logits = torch.rand(2, 10, requires_grad=True)

        loss = matryoshka_cross_entropy(
            {4: zero_weight_logits, 8: weighted_logits}, labels, [0.0, 1.0]
        )

        self.assertTrue(torch.allclose(loss, F.cross_entropy(weighted_logits, labels)))
        loss.backward()
        self.assertIsNone(zero_weight_logits.grad)
        self.assertIsNotNone(weighted_logits.grad)

    def test_loss_with_all_zero_weights_preserves_autograd(self) -> None:
        logits_by_dimension = {
            4: torch.rand(2, 10, requires_grad=True),
            8: torch.rand(2, 10, requires_grad=True),
        }

        loss = matryoshka_cross_entropy(logits_by_dimension, torch.tensor([0, 1]), [0.0, 0.0])

        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for logits in logits_by_dimension.values():
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_loss_with_all_zero_weights_ignores_nan_logits(self) -> None:
        logits_by_dimension = {
            4: torch.full((2, 10), float("nan"), requires_grad=True),
            8: torch.full((2, 10), float("nan"), requires_grad=True),
        }

        loss = matryoshka_cross_entropy(logits_by_dimension, torch.tensor([0, 1]), [0.0, 0.0])

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        for logits in logits_by_dimension.values():
            self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_loss_rejects_invalid_logit_shapes(self) -> None:
        labels = torch.tensor([0, 1, 2])
        for logits_by_dimension in (
            {4: torch.rand(2, 10)},
            {4: torch.rand(3, 9)},
        ):
            with self.subTest(logits_by_dimension=logits_by_dimension):
                with self.assertRaises(ValueError):
                    matryoshka_cross_entropy(logits_by_dimension, labels, [1.0])

    def test_loss_rejects_integer_logits_with_value_error(self) -> None:
        with self.assertRaises(ValueError):
            matryoshka_cross_entropy(
                {4: torch.ones(2, 10, dtype=torch.long)}, torch.tensor([0, 1]), [1.0]
            )

    def test_loss_rejects_logits_on_a_different_device_with_value_error(self) -> None:
        with self.assertRaises(ValueError):
            matryoshka_cross_entropy(
                {4: torch.empty(2, 10, device="meta")}, torch.tensor([0, 1]), [1.0]
            )

    def test_loss_rejects_nonzero_logits_with_mixed_dtypes(self) -> None:
        with self.assertRaises(ValueError):
            matryoshka_cross_entropy(
                {4: torch.rand(2, 10), 8: torch.rand(2, 10, dtype=torch.float64)},
                torch.tensor([0, 1]),
                [1.0, 1.0],
            )

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
                    matryoshka_cross_entropy(logits_by_dimension, labels, [1.0])

    def test_loss_rejects_invalid_relative_importance_with_value_error(self) -> None:
        logits_by_dimension = {4: torch.rand(2, 10), 8: torch.rand(2, 10)}
        labels = torch.tensor([0, 1])
        for relative_importance in (
            [],
            [1.0],
            [1.0, 1.0, 1.0],
            [1.0, -0.5],
            [1.0, float("nan")],
        ):
            with self.subTest(relative_importance=relative_importance):
                with self.assertRaises(ValueError):
                    matryoshka_cross_entropy(logits_by_dimension, labels, relative_importance)


if __name__ == "__main__":
    unittest.main()