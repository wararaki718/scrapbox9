import math

import torch


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