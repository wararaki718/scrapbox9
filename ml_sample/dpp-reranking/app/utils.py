import sys
from collections.abc import Sequence
from typing import TextIO

from app.data import Recommendation


def show(
    ordinary: Sequence[Recommendation],
    reranked: Sequence[Recommendation],
    stream: TextIO | None = None,
) -> None:
    output = sys.stdout if stream is None else stream
    for title, recommendations in (
        ("Relevance ranking", ordinary),
        ("DPP ranking", reranked),
    ):
        print(title, file=output)
        print("rank  item  category  score", file=output)
        for rank, recommendation in enumerate(recommendations, start=1):
            print(
                f"{rank:>4}  {recommendation.item_id:>4}  "
                f"{recommendation.category:<8}  {recommendation.score:.4f}",
                file=output,
            )
        print(file=output)