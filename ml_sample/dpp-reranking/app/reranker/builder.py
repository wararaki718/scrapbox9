import math

import torch
from torch.nn import functional


class KernelBuilder:
    def __init__(self, quality_scale: float = 1.0, epsilon: float = 1e-6) -> None:
        if not math.isfinite(quality_scale) or quality_scale < 0:
            raise ValueError("quality_scale must be finite and non-negative")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.quality_scale = quality_scale
        self.epsilon = epsilon

    def build(self, scores: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 1 or item_embeddings.ndim != 2:
            raise ValueError("scores and item_embeddings must have ranks 1 and 2")
        if scores.shape[0] != item_embeddings.shape[0]:
            raise ValueError("candidate counts must match")
        if not torch.isfinite(scores).all() or not torch.isfinite(item_embeddings).all():
            raise ValueError("scores and item_embeddings must be finite")

        normalized = functional.normalize(item_embeddings, dim=1)
        similarity = normalized @ normalized.T
        quality = torch.exp(self.quality_scale * scores)
        kernel = quality[:, None] * similarity * quality[None, :]
        identity = torch.eye(kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
        return kernel + self.epsilon * identity