from dataclasses import dataclass

from app.schemas.interaction import Interaction
from app.schemas.item import Item


@dataclass(frozen=True)
class SampleData:
    num_users: int
    items: tuple[Item, ...]
    interactions: tuple[Interaction, ...]