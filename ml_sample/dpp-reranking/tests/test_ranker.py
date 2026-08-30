import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models import TwoTowerModel
from app.ranker import CandidateRanker
from app.schemas import Interaction, Item, SampleData


class CandidateRankerTests(unittest.TestCase):
    def test_excludes_seen_items_and_returns_aligned_relevance_order(self) -> None:
        model = TwoTowerModel(num_users=1, num_items=3, embedding_dim=2)
        with torch.no_grad():
            model.user_tower.embedding.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.item_tower.embedding.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
            )
        data = SampleData(
            num_users=1,
            items=(Item(0, "books"), Item(1, "music"), Item(2, "sports")),
            interactions=(Interaction(0, 0),),
        )

        ranking = CandidateRanker().rank(model, data, user_id=0)

        self.assertEqual([row.item_id for row in ranking.recommendations], [1, 2])
        self.assertTrue(torch.all(ranking.scores[:-1] >= ranking.scores[1:]))
        self.assertEqual(ranking.item_embeddings.shape, (2, 2))
        self.assertAlmostEqual(ranking.recommendations[0].score, ranking.scores[0].item())

    def test_rejects_user_without_unseen_candidates(self) -> None:
        model = TwoTowerModel(num_users=1, num_items=1, embedding_dim=2)
        data = SampleData(1, (Item(0, "books"),), (Interaction(0, 0),))

        with self.assertRaisesRegex(ValueError, "unseen candidates"):
            CandidateRanker().rank(model, data, user_id=0)