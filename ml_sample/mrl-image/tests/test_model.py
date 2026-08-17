import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model import MatryoshkaImageClassifier


class MatryoshkaImageClassifierTests(unittest.TestCase):
    def test_forward_returns_logits_for_each_requested_prefix(self) -> None:
        model = MatryoshkaImageClassifier(embedding_dim=16, dimensions=[4, 8, 16])

        logits_by_dimension = model(torch.rand(3, 3, 32, 32))

        self.assertEqual(list(logits_by_dimension), [4, 8, 16])
        self.assertEqual(set(logits_by_dimension), {4, 8, 16})
        for dimension in (4, 8, 16):
            with self.subTest(dimension=dimension):
                self.assertEqual(logits_by_dimension[dimension].shape, (3, 10))

    def test_state_dict_registers_classifier_parameters_for_each_prefix(self) -> None:
        model = MatryoshkaImageClassifier(embedding_dim=16, dimensions=[4, 8, 16])

        state_dict = model.state_dict()

        for dimension in (4, 8, 16):
            with self.subTest(dimension=dimension):
                self.assertIn(f"classifiers.{dimension}.weight", state_dict)
                self.assertIn(f"classifiers.{dimension}.bias", state_dict)

    def test_constructor_rejects_invalid_dimensions(self) -> None:
        for embedding_dim, dimensions in (
            (0, [4]),
            (16, []),
            (16, [0]),
            (16, [-1, 4]),
            (16, [17]),
            (16, [4, 4]),
        ):
            with self.subTest(embedding_dim=embedding_dim, dimensions=dimensions):
                with self.assertRaises(ValueError):
                    MatryoshkaImageClassifier(embedding_dim, dimensions)


if __name__ == "__main__":
    unittest.main()