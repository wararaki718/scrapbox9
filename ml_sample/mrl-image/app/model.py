from collections.abc import Sequence

from torch import Tensor, nn


class MatryoshkaImageClassifier(nn.Module):
    def __init__(self, embedding_dim: int, dimensions: Sequence[int]) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.dimensions = tuple(dimensions)
        if not self.dimensions:
            raise ValueError("dimensions must not be empty")
        if len(set(self.dimensions)) != len(self.dimensions):
            raise ValueError("dimensions must not contain duplicates")
        if any(dimension <= 0 or dimension > embedding_dim for dimension in self.dimensions):
            raise ValueError("each dimension must be positive and no greater than embedding_dim")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(64, embedding_dim)
        self.classifiers = nn.ModuleDict(
            {str(dimension): nn.Linear(dimension, 10) for dimension in self.dimensions}
        )

    def forward(self, images: Tensor) -> dict[int, Tensor]:
        features = self.features(images).flatten(start_dim=1)
        embedding = self.projection(features)
        return {
            dimension: self.classifiers[str(dimension)](embedding[:, :dimension])
            for dimension in self.dimensions
        }