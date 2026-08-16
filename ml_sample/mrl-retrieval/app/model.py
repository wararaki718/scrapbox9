import torch
from torch import nn
from torch.nn import functional as functional


class MatryoshkaEncoder(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        max_length: int,
    ) -> None:
        super().__init__()
        self._embedding_dim = embedding_dim
        self._token_embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self._position_embedding = nn.Embedding(max_length, embedding_dim)

        layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, batch_first=True)
        self._encoder = nn.TransformerEncoder(layer, num_layers)
        self._projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        dimension: int | None = None,
    ) -> torch.Tensor:
        dimension = self._embedding_dim if dimension is None else dimension
        if not 0 < dimension <= self._embedding_dim:
            raise ValueError("dimension must be within embedding width")

        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        hidden = self._encoder(
            self._token_embedding(input_ids) + self._position_embedding(positions),
            src_key_padding_mask=attention_mask == 0,
        )
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1)
        return functional.normalize(self._projection(pooled)[:, :dimension], dim=1)
    