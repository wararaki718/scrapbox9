import torch

from app.config import TrainConfig
from app.data import create_reranking_data, create_training_data
from app.models import TwoTowerModel
from app.ranker import CandidateRanker
from app.reranker import DPPReranker
from app.train import train
from app.utils import show


def main() -> None:
    config = TrainConfig()
    torch.manual_seed(config.seed)
    training_data = create_training_data()
    reranking_data = create_reranking_data()
    model = TwoTowerModel(
        training_data.num_users,
        len(training_data.items),
        config.embedding_dim,
    )
    train(
        model,
        training_data.interactions,
        len(training_data.items),
        config,
    )

    ranking = CandidateRanker().rank(model, reranking_data, user_id=0)
    top_k = 20
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