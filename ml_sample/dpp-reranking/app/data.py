from app.schemas import Interaction, Item, SampleData


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