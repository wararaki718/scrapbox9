from dataclasses import dataclass

import torch

from app.schemas.recommendation import Recommendation


@dataclass(frozen=True)
class CandidateRanking:
    recommendations: tuple[Recommendation, ...]
    scores: torch.Tensor
    item_embeddings: torch.Tensor