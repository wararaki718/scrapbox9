import sys
from collections.abc import Sequence
from typing import TextIO

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