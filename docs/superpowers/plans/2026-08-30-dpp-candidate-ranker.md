# DPP Candidate Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move candidate filtering, relevance ranking, and recommendation construction from `main.py` into `CandidateRanker`.

**Architecture:** `CandidateRanker.rank()` accepts a trained two-tower model, sample data, and user ID, then returns an immutable `CandidateRanking` whose recommendations, scores, and item embeddings share one descending relevance order. `main.py` retains only workflow composition.

**Tech Stack:** Python 3.11, PyTorch, dataclasses, unittest

---

### Task 1: Add Candidate Ranking Result And Ranker

**Files:**
- Create: `ml_sample/dpp-reranking/app/schemas/candidate_ranking.py`
- Modify: `ml_sample/dpp-reranking/app/schemas/__init__.py`
- Create: `ml_sample/dpp-reranking/app/ranker.py`
- Create: `ml_sample/dpp-reranking/tests/test_ranker.py`
- Modify: `ml_sample/dpp-reranking/app/main.py`

- [ ] **Step 1: Write failing ranker tests**

```python
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.models import TwoTowerModel
from app.ranker import CandidateRanker
from app.schemas import Interaction, Item, SampleData


class CandidateRankerTests(unittest.TestCase):
    def test_excludes_seen_items_and_returns_aligned_relevance_order(self) -> None:
        model = TwoTowerModel(num_users=1, num_items=3, embedding_dim=2)
        with torch.no_grad():
            model.user_tower.embedding.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.item_tower.embedding.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
            )
        data = SampleData(
            num_users=1,
            items=(Item(0, "books"), Item(1, "music"), Item(2, "sports")),
            interactions=(Interaction(0, 0),),
        )

        ranking = CandidateRanker().rank(model, data, user_id=0)

        self.assertEqual([row.item_id for row in ranking.recommendations], [1, 2])
        self.assertTrue(torch.all(ranking.scores[:-1] >= ranking.scores[1:]))
        self.assertEqual(ranking.item_embeddings.shape, (2, 2))
        self.assertAlmostEqual(ranking.recommendations[0].score, ranking.scores[0].item())

    def test_rejects_user_without_unseen_candidates(self) -> None:
        model = TwoTowerModel(num_users=1, num_items=1, embedding_dim=2)
        data = SampleData(1, (Item(0, "books"),), (Interaction(0, 0),))

        with self.assertRaisesRegex(ValueError, "unseen candidates"):
            CandidateRanker().rank(model, data, user_id=0)
```

- [ ] **Step 2: Run tests and verify the missing-module failure**

Run: `cd ml_sample/dpp-reranking && python -m unittest tests.test_ranker -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.ranker'`.

- [ ] **Step 3: Add the immutable ranking result**

```python
# app/schemas/candidate_ranking.py
from dataclasses import dataclass

import torch

from app.schemas.recommendation import Recommendation


@dataclass(frozen=True)
class CandidateRanking:
    recommendations: tuple[Recommendation, ...]
    scores: torch.Tensor
    item_embeddings: torch.Tensor
```

Export `CandidateRanking` from `app/schemas/__init__.py` and add it to
`__all__`.

- [ ] **Step 4: Implement `CandidateRanker`**

```python
# app/ranker.py
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
```

- [ ] **Step 5: Run focused tests and verify they pass**

Run: `cd ml_sample/dpp-reranking && python -m unittest tests.test_ranker -v`

Expected: 2 tests pass.

- [ ] **Step 6: Replace ranking logic in `main.py`**

Import `CandidateRanker`, remove the direct `Recommendation` and `torch`
imports, then replace candidate filtering through recommendation construction
with:

```python
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
```

Keep `torch.manual_seed(config.seed)` by importing `torch` in `main.py`.

- [ ] **Step 7: Run complete verification and commit**

Run: `cd ml_sample/dpp-reranking && make test && make run`

Expected: all 22 tests pass, followed by relevance and DPP ranking tables.

```bash
git add ml_sample/dpp-reranking/app ml_sample/dpp-reranking/tests/test_ranker.py
git commit -m "refactor: extract candidate ranker"
```