import sys
from collections.abc import Sequence
from typing import TextIO

from app.evaluate import EvaluationResult
from app.schemas import Recommendation


def show(
    ordinary: Sequence[Recommendation],
    reranked: Sequence[Recommendation],
    stream: TextIO | None = None,
) -> None:
    output = sys.stdout if stream is None else stream
    reranked_item_ids = {recommendation.item_id for recommendation in reranked}
    for title, recommendations in (
        ("Relevance ranking", ordinary),
        ("DPP ranking", reranked),
    ):
        print(title, file=output)
        print("rank  item  category  score   label", file=output)
        for rank, recommendation in enumerate(recommendations, start=1):
            label = (
                "reranked" if recommendation.item_id in reranked_item_ids else "-"
            )
            print(
                f"{rank:>4}  {recommendation.item_id:>4}  "
                f"{recommendation.category:<8}  {recommendation.score:.4f}  {label}",
                file=output,
            )
        print(file=output)


def show_evaluation(
    results: Sequence[EvaluationResult],
    stream: TextIO | None = None,
) -> None:
    if not results:
        raise ValueError("results must not be empty")

    output = sys.stdout if stream is None else stream
    cutoffs = tuple(cutoff for cutoff, _ in results[0].ndcg_by_cutoff)
    print("Offline evaluation", file=output)
    ndcg_header = "  ".join(f"NDCG@{cutoff}" for cutoff in cutoffs)
    diversity_cutoff = max(cutoffs)
    print(
        f"model        {ndcg_header}  ILD@{diversity_cutoff}  "
        f"coverage@{diversity_cutoff}",
        file=output,
    )
    for result in results:
        ndcg_values = "  ".join(
            f"{value:.4f}" for _, value in result.ndcg_by_cutoff
        )
        print(
            f"{result.name:<12} {ndcg_values}  "
            f"{result.intra_list_diversity:.4f}  {result.category_coverage}",
            file=output,
        )
    print(file=output)