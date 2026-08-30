import torch

from app.config import TrainConfig
from app.data import create_sample_data
from app.models import TwoTowerModel
from app.reranker import DPPReranker
from app.schemas import Recommendation
from app.train import train
from app.utils import show


def main() -> None:
    config = TrainConfig()
    torch.manual_seed(config.seed)
    data = create_sample_data()
    model = TwoTowerModel(data.num_users, len(data.items), config.embedding_dim)
    train(model, data.interactions, len(data.items), config)

    user_id = 0
    seen = {row.item_id for row in data.interactions if row.user_id == user_id}
    candidate_ids = [item.item_id for item in data.items if item.item_id not in seen]
    item_by_id = {item.item_id: item for item in data.items}

    model.eval()
    with torch.no_grad():
        item_ids = torch.tensor(candidate_ids)
        user_ids = torch.full_like(item_ids, user_id)
        scores = model(user_ids, item_ids)
        order = torch.argsort(scores, descending=True)
        sorted_ids = item_ids[order]
        sorted_scores = scores[order]
        sorted_embeddings = model.item_tower(sorted_ids)

    candidates = [
        Recommendation(
            item_id=int(item_id.item()),
            category=item_by_id[int(item_id.item())].category,
            score=float(score.item()),
        )
        for item_id, score in zip(sorted_ids, sorted_scores, strict=True)
    ]
    top_k = 5
    selected = DPPReranker().rerank(sorted_scores, sorted_embeddings, top_k)
    show(candidates[:top_k], [candidates[index] for index in selected])


if __name__ == "__main__":
    main()