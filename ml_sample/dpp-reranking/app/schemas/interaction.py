from dataclasses import dataclass


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int