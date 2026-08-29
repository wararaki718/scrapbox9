import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    embedding_dim: int = 8
    epochs: int = 100
    learning_rate: float = 0.05
    seed: int = 7

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")