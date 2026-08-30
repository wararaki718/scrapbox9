import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.evaluate import Evaluator
from app.schemas import CandidateRanking, Recommendation


class RecordingReranker:
    def __init__(self) -> None:
        self.candidate_count = 0
        self.selection_count = 0

    def rerank(
        self,
        scores: torch.Tensor,
        item_embeddings: torch.Tensor,
        top_k: int,
    ) -> list[int]:
        self.candidate_count = len(scores)
        self.selection_count = top_k
        return list(range(26, 6, -1))


class EvaluatorTests(unittest.TestCase):
    def test_compares_relevance_with_reordered_dpp_selection(self) -> None:
        recommendations = tuple(
            Recommendation(item_id, ("books", "music", "sports")[item_id % 3], 1.0)
            for item_id in range(27)
        )
        ranking = CandidateRanking(
            recommendations,
            torch.linspace(1.0, 0.0, 27),
            torch.tensor(
                [[1.0, float(item_id % 3)] for item_id in range(27)]
            ),
        )
        reranker = RecordingReranker()

        results = Evaluator(reranker).evaluate(
            ranking,
            {item_id: float(3 - item_id % 4) for item_id in range(27)},
        )

        self.assertEqual(reranker.candidate_count, 27)
        self.assertEqual(reranker.selection_count, 20)
        self.assertEqual([result.name for result in results], ["Reranking 1", "DPP"])
        self.assertEqual(results[0].item_ids, tuple(range(20)))
        self.assertEqual(results[1].item_ids, tuple(range(7, 27)))
        for result in results:
            self.assertEqual(dict(result.ndcg_by_cutoff).keys(), {5, 10, 20})
            self.assertGreaterEqual(result.intra_list_diversity, 0.0)
            self.assertEqual(result.category_coverage, 3)


if __name__ == "__main__":
    unittest.main()