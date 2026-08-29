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


class GreedyMapSelector:
    def __init__(self, tolerance: float = 1e-10) -> None:
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and non-negative")
        self.tolerance = tolerance

    def select(self, kernel: torch.Tensor, top_k: int) -> list[int]:
        if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
            raise ValueError("kernel must be square")
        if not torch.isfinite(kernel).all():
            raise ValueError("kernel must be finite")
        if not torch.allclose(kernel, kernel.T, atol=1e-6):
            raise ValueError("kernel must be symmetric")

        candidate_count = kernel.shape[0]
        if top_k <= 0 or top_k > candidate_count:
            raise ValueError("top_k must be within the candidate count")

        coefficients = torch.zeros(
            (top_k, candidate_count), dtype=kernel.dtype, device=kernel.device
        )
        residuals = torch.diagonal(kernel).clone()
        selected: list[int] = []

        for iteration in range(top_k):
            item_index = int(torch.argmax(residuals).item())
            if residuals[item_index] <= self.tolerance:
                break
            selected.append(item_index)
            if iteration == top_k - 1:
                break

            previous = coefficients[:iteration, item_index]
            projection = previous @ coefficients[:iteration]
            update = (kernel[item_index] - projection) / torch.sqrt(residuals[item_index])
            coefficients[iteration] = update
            residuals = torch.clamp(residuals - update.square(), min=0)
            residuals[selected] = -torch.inf

        return selected


class DPPReranker:
    def __init__(
        self,
        kernel_builder: KernelBuilder | None = None,
        selector: GreedyMapSelector | None = None,
    ) -> None:
        self.kernel_builder = kernel_builder or KernelBuilder()
        self.selector = selector or GreedyMapSelector()

    def rerank(
        self,
        scores: torch.Tensor,
        item_embeddings: torch.Tensor,
        top_k: int,
    ) -> list[int]:
        kernel = self.kernel_builder.build(scores, item_embeddings)
        return self.selector.select(kernel, top_k)