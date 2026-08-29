from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    item_id: int
    category: str


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    category: str
    score: float


@dataclass(frozen=True)
class SampleData:
    num_users: int
    items: tuple[Item, ...]
    interactions: tuple[Interaction, ...]


def create_sample_data() -> SampleData:
    categories = ("books", "music", "sports")
    items = tuple(Item(item_id, categories[item_id // 4]) for item_id in range(12))
    interactions = tuple(
        Interaction(user_id, item_id)
        for user_id, item_ids in (
            (0, (0, 1, 4)),
            (1, (2, 3, 8)),
            (2, (5, 6, 9)),
        )
        for item_id in item_ids
    )
    return SampleData(num_users=3, items=items, interactions=interactions)