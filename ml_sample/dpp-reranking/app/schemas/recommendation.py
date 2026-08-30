from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    category: str
    score: float