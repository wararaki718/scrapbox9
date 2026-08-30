import math
from collections.abc import Mapping, Sequence


def _dcg(relevances: Sequence[float]) -> float:
    return sum(
        (2.0**relevance - 1.0) / math.log2(rank + 2)
        for rank, relevance in enumerate(relevances)
    )


def ndcg_at_k(
    item_ids: Sequence[int],
    relevance_by_item_id: Mapping[int, float],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if any(
        not math.isfinite(relevance) or relevance < 0
        for relevance in relevance_by_item_id.values()
    ):
        raise ValueError("relevance judgments must be finite and non-negative")

    actual = [relevance_by_item_id.get(item_id, 0.0) for item_id in item_ids[:k]]
    ideal = sorted(relevance_by_item_id.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(actual) / ideal_dcg