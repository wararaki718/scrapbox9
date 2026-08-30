from collections.abc import Mapping, Sequence


def category_coverage_at_k(
    item_ids: Sequence[int],
    category_by_item_id: Mapping[int, str],
    k: int,
) -> int:
    if k <= 0:
        raise ValueError("k must be positive")

    selected_item_ids = item_ids[:k]
    try:
        return len({category_by_item_id[item_id] for item_id in selected_item_ids})
    except KeyError as error:
        raise ValueError("all selected items must have categories") from error