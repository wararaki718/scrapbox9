import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.models import TwoTowerModel


class ModelTests(unittest.TestCase):
    def test_maps_cosine_similarity_to_zero_one_range(self) -> None:
        model = TwoTowerModel(num_users=1, num_items=2, embedding_dim=2)
        with torch.no_grad():
            model.user_tower.embedding.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.item_tower.embedding.weight.copy_(
                torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
            )

        scores = model(torch.tensor([0, 0]), torch.tensor([0, 1]))

        self.assertTrue(torch.allclose(scores, torch.tensor([1.0, 0.0])))

    def test_scores_pairs_and_updates_both_towers(self) -> None:
        model = TwoTowerModel(num_users=2, num_items=3, embedding_dim=4)
        scores = model(torch.tensor([0, 1]), torch.tensor([1, 2]))

        self.assertEqual(scores.shape, (2,))
        scores.sum().backward()
        self.assertIsNotNone(model.user_tower.embedding.weight.grad)
        self.assertIsNotNone(model.item_tower.embedding.weight.grad)


if __name__ == "__main__":
    unittest.main()