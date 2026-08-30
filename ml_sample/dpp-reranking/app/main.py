import torch

from app.config import TrainConfig
from app.data import create_reranking_data, create_training_data
from app.evaluate import Evaluator
from app.models import TwoTowerModel
from app.ranker import CandidateRanker
from app.reranker import DPPReranker
from app.train import train
from app.utils import show, show_evaluation


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
    candidate_count = 20
    rerank_count = 10
    selected = DPPReranker().rerank(
        ranking.scores[:candidate_count],
        ranking.item_embeddings[:candidate_count],
        rerank_count,
    )
    show(
        ranking.recommendations[:candidate_count],
        [ranking.recommendations[index] for index in selected],
    )
    relevance_by_category = {"books": 3.0, "music": 2.0, "sports": 1.0}
    relevance_by_item_id = {
        recommendation.item_id: relevance_by_category[recommendation.category]
        for recommendation in ranking.recommendations
    }
    show_evaluation(Evaluator().evaluate(ranking, relevance_by_item_id))


if __name__ == "__main__":
    main()