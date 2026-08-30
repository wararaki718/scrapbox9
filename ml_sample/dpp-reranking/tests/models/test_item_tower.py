import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.models import ItemTower


class ItemTowerTests(unittest.TestCase):
    def test_returns_unit_normalized_embeddings(self) -> None:
        output = ItemTower(3, 4)(torch.tensor([0, 2]))

        self.assertEqual(output.shape, (2, 4))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_rejects_non_positive_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            ItemTower(0, 4)
        with self.assertRaises(ValueError):
            ItemTower(3, 0)


if __name__ == "__main__":
    unittest.main()