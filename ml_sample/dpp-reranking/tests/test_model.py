import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models import TwoTowerModel


class ModelTests(unittest.TestCase):
    def test_scores_pairs_and_updates_both_towers(self) -> None:
        model = TwoTowerModel(num_users=2, num_items=3, embedding_dim=4)
        scores = model(torch.tensor([0, 1]), torch.tensor([1, 2]))

        self.assertEqual(scores.shape, (2,))
        scores.sum().backward()
        self.assertIsNotNone(model.user_tower.embedding.weight.grad)
        self.assertIsNotNone(model.item_tower.embedding.weight.grad)