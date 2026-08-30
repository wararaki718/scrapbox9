from app.schemas import Interaction, Item, SampleData


def _create_items() -> tuple[Item, ...]:
    categories = ("books", "music", "sports")
    return tuple(Item(item_id, categories[item_id // 10]) for item_id in range(30))


def create_training_data() -> SampleData:
    interactions = tuple(
        Interaction(user_id, item_id)
        for user_id, item_ids in (
            (0, (0, 1, 10)),
            (1, (2, 11, 20)),
            (2, (3, 12, 21)),
        )
        for item_id in item_ids
    )
    return SampleData(num_users=3, items=_create_items(), interactions=interactions)


def create_reranking_data() -> SampleData:
    interactions = tuple(
        Interaction(user_id, item_id)
        for user_id, item_ids in (
            (0, (4, 13, 22)),
            (1, (5, 14, 23)),
            (2, (6, 15, 24)),
        )
        for item_id in item_ids
    )
    return SampleData(num_users=3, items=_create_items(), interactions=interactions)