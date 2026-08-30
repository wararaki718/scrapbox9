import torch

from app.models import TwoTowerModel
from app.schemas import CandidateRanking, Recommendation, SampleData


class CandidateRanker:
    def rank(
        self,
        model: TwoTowerModel,
        data: SampleData,
        user_id: int,
    ) -> CandidateRanking:
        seen = {row.item_id for row in data.interactions if row.user_id == user_id}
        candidate_ids = [item.item_id for item in data.items if item.item_id not in seen]
        if not candidate_ids:
            raise ValueError("user must have unseen candidates")
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

        recommendations = tuple(
            Recommendation(
                item_id=int(item_id.item()),
                category=item_by_id[int(item_id.item())].category,
                score=float(score.item()),
            )
            for item_id, score in zip(sorted_ids, sorted_scores, strict=True)
        )
        return CandidateRanking(recommendations, sorted_scores, sorted_embeddings)