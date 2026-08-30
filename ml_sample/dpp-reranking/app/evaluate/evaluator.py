from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from app.evaluate.metrics import (
    category_coverage_at_k,
    intra_list_diversity_at_k,
    ndcg_at_k,
)
from app.reranker import DPPReranker
from app.schemas import CandidateRanking


class Reranker(Protocol):
    def rerank(
        self,
        scores: torch.Tensor,
        item_embeddings: torch.Tensor,
        top_k: int,
    ) -> list[int]: ...


@dataclass(frozen=True)
class EvaluationResult:
    name: str
    item_ids: tuple[int, ...]
    ndcg_by_cutoff: tuple[tuple[int, float], ...]
    intra_list_diversity: float
    category_coverage: int


class Evaluator:
    def __init__(self, reranker: Reranker | None = None) -> None:
        self.reranker = reranker or DPPReranker()

    def evaluate(
        self,
        ranking: CandidateRanking,
        relevance_by_item_id: Mapping[int, float],
        selection_count: int = 20,
        cutoffs: Sequence[int] = (5, 10, 20),
    ) -> tuple[EvaluationResult, EvaluationResult]:
        candidate_count = len(ranking.recommendations)
        if ranking.scores.shape != (candidate_count,):
            raise ValueError("scores must align with recommendations")
        if ranking.item_embeddings.ndim != 2 or ranking.item_embeddings.shape[0] != candidate_count:
            raise ValueError("item embeddings must align with recommendations")
        if not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
            raise ValueError("cutoffs must be positive")
        if selection_count < max(cutoffs) or selection_count > candidate_count:
            raise ValueError("selection count must cover cutoffs and candidates")

        selected_indices = self.reranker.rerank(
            ranking.scores,
            ranking.item_embeddings,
            selection_count,
        )
        if len(selected_indices) != selection_count or len(set(selected_indices)) != selection_count:
            raise ValueError("reranker must return the requested number of unique indices")
        if any(index < 0 or index >= candidate_count for index in selected_indices):
            raise ValueError("reranker returned an invalid index")

        relevance_item_ids = tuple(
            recommendation.item_id
            for recommendation in ranking.recommendations[:selection_count]
        )
        dpp_item_ids = tuple(
            ranking.recommendations[index].item_id for index in sorted(selected_indices)
        )
        embedding_by_item_id = {
            recommendation.item_id: embedding
            for recommendation, embedding in zip(
                ranking.recommendations,
                ranking.item_embeddings,
                strict=True,
            )
        }
        category_by_item_id = {
            recommendation.item_id: recommendation.category
            for recommendation in ranking.recommendations
        }

        return tuple(
            self._evaluate_ranking(
                name,
                item_ids,
                relevance_by_item_id,
                embedding_by_item_id,
                category_by_item_id,
                cutoffs,
            )
            for name, item_ids in (
                ("Reranking 1", relevance_item_ids),
                ("DPP", dpp_item_ids),
            )
        )

    @staticmethod
    def _evaluate_ranking(
        name: str,
        item_ids: tuple[int, ...],
        relevance_by_item_id: Mapping[int, float],
        embedding_by_item_id: Mapping[int, torch.Tensor],
        category_by_item_id: Mapping[int, str],
        cutoffs: Sequence[int],
    ) -> EvaluationResult:
        diversity_cutoff = max(cutoffs)
        return EvaluationResult(
            name=name,
            item_ids=item_ids,
            ndcg_by_cutoff=tuple(
                (cutoff, ndcg_at_k(item_ids, relevance_by_item_id, cutoff))
                for cutoff in cutoffs
            ),
            intra_list_diversity=intra_list_diversity_at_k(
                item_ids, embedding_by_item_id, diversity_cutoff
            ),
            category_coverage=category_coverage_at_k(
                item_ids, category_by_item_id, diversity_cutoff
            ),
        )