import torch

from app.reranker.builder import KernelBuilder
from app.reranker.selector import GreedyMapSelector


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