import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.item_tower import ItemTower
from app.user_tower import UserTower


class TowerTests(unittest.TestCase):
    def test_towers_return_unit_normalized_embeddings(self) -> None:
        ids = torch.tensor([0, 2])
        for tower in (UserTower(3, 4), ItemTower(3, 4)):
            with self.subTest(tower=type(tower).__name__):
                output = tower(ids)
                self.assertEqual(output.shape, (2, 4))
                self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_towers_reject_non_positive_dimensions(self) -> None:
        for tower_type in (UserTower, ItemTower):
            with self.subTest(tower=tower_type.__name__), self.assertRaises(ValueError):
                tower_type(0, 4)
            with self.subTest(tower=tower_type.__name__), self.assertRaises(ValueError):
                tower_type(3, 0)