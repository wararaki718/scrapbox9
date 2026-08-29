import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import TrainConfig
from app.data import Interaction
from app.model import TwoTowerModel
from app.train import train


class TrainTests(unittest.TestCase):
    def test_bpr_training_lowers_loss_and_ranks_positive_higher(self) -> None:
        torch.manual_seed(0)
        model = TwoTowerModel(1, 2, 4)
        losses = train(
            model,
            (Interaction(0, 0),),
            num_items=2,
            config=TrainConfig(embedding_dim=4, epochs=30, learning_rate=0.05, seed=3),
        )

        self.assertLess(losses[-1], losses[0])
        with torch.no_grad():
            scores = model(torch.tensor([0, 0]), torch.tensor([0, 1]))
        self.assertGreater(scores[0].item(), scores[1].item())

    def test_training_rejects_user_with_no_negative_item(self) -> None:
        model = TwoTowerModel(1, 1, 2)
        with self.assertRaisesRegex(ValueError, "negative item"):
            train(model, (Interaction(0, 0),), 1, TrainConfig(embedding_dim=2))