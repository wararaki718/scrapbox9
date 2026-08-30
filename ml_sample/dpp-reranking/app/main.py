import torch

from app.config import TrainConfig
from app.data import create_sample_data
from app.models import TwoTowerModel
from app.ranker import CandidateRanker
from app.reranker import DPPReranker
from app.train import train
from app.utils import show


def main() -> None:
    config = TrainConfig()
    torch.manual_seed(config.seed)
    data = create_sample_data()
    model = TwoTowerModel(data.num_users, len(data.items), config.embedding_dim)
    train(model, data.interactions, len(data.items), config)

    ranking = CandidateRanker().rank(model, data, user_id=0)
    top_k = 5
    selected = DPPReranker().rerank(
        ranking.scores,
        ranking.item_embeddings,
        top_k,
    )
    show(
        ranking.recommendations[:top_k],
        [ranking.recommendations[index] for index in selected],
    )


if __name__ == "__main__":
    main()