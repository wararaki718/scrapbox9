# DPP Reranking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible PyTorch sample that trains a two-tower recommender and applies Fast Greedy MAP DPP reranking to its candidates.

**Architecture:** Separate user and item embedding towers are composed by a two-tower model and trained with BPR loss on deterministic implicit-feedback data. `DPPReranker` composes a quality-weighted `KernelBuilder` and an incremental-Cholesky `GreedyMapSelector`; the CLI converts selected indices into structured recommendations and delegates all output to `utils.show`.

**Tech Stack:** Python 3.11, PyTorch 2.x, standard-library `unittest`, Make

---

## File Structure

- `ml_sample/dpp-reranking/app/config.py`: immutable training configuration.
- `ml_sample/dpp-reranking/app/data.py`: sample entities, recommendation value object, and deterministic data.
- `ml_sample/dpp-reranking/app/user_tower.py`: normalized user embedding tower.
- `ml_sample/dpp-reranking/app/item_tower.py`: normalized item embedding tower.
- `ml_sample/dpp-reranking/app/model.py`: two-tower composition and dot-product scoring.
- `ml_sample/dpp-reranking/app/train.py`: deterministic negative sampling and BPR optimization.
- `ml_sample/dpp-reranking/app/reranker.py`: kernel construction, Fast Greedy MAP selection, and reranker facade.
- `ml_sample/dpp-reranking/app/utils.py`: recommendation-list display.
- `ml_sample/dpp-reranking/app/main.py`: end-to-end demo orchestration.
- `ml_sample/dpp-reranking/tests/`: one focused `unittest` module per application responsibility.
- `ml_sample/dpp-reranking/Makefile`: run and test entry points.
- `ml_sample/dpp-reranking/requirements.txt`: PyTorch dependency.
- `ml_sample/dpp-reranking/README.md`: setup, algorithm, and usage documentation.

### Task 1: Configuration And Sample Data

**Files:**
- Create: `ml_sample/dpp-reranking/app/__init__.py`
- Create: `ml_sample/dpp-reranking/app/config.py`
- Create: `ml_sample/dpp-reranking/app/data.py`
- Create: `ml_sample/dpp-reranking/tests/test_data.py`

- [ ] **Step 1: Write the failing configuration and data tests**

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import TrainConfig
from app.data import Recommendation, create_sample_data


class DataTests(unittest.TestCase):
    def test_sample_data_has_contiguous_ids_and_valid_interactions(self) -> None:
        data = create_sample_data()

        self.assertEqual([item.item_id for item in data.items], list(range(len(data.items))))
        self.assertGreaterEqual(len({item.category for item in data.items}), 3)
        self.assertTrue(all(0 <= row.user_id < data.num_users for row in data.interactions))
        self.assertTrue(all(0 <= row.item_id < len(data.items) for row in data.interactions))

    def test_recommendation_is_immutable(self) -> None:
        recommendation = Recommendation(1, "books", 0.5)
        with self.assertRaises(AttributeError):
            recommendation.score = 1.0  # type: ignore[misc]

    def test_train_config_rejects_invalid_values(self) -> None:
        for kwargs in (
            {"embedding_dim": 0},
            {"epochs": 0},
            {"learning_rate": 0.0},
            {"learning_rate": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                TrainConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify the missing-module failure**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_data.py' -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.config'`.

- [ ] **Step 3: Implement the immutable configuration and sample data**

```python
# app/config.py
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    embedding_dim: int = 8
    epochs: int = 100
    learning_rate: float = 0.05
    seed: int = 7

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
```

```python
# app/data.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    item_id: int
    category: str


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    category: str
    score: float


@dataclass(frozen=True)
class SampleData:
    num_users: int
    items: tuple[Item, ...]
    interactions: tuple[Interaction, ...]


def create_sample_data() -> SampleData:
    categories = ("books", "music", "sports")
    items = tuple(
        Item(item_id, categories[item_id // 4])
        for item_id in range(12)
    )
    interactions = tuple(
        Interaction(user_id, item_id)
        for user_id, item_ids in (
            (0, (0, 1, 4)),
            (1, (2, 3, 8)),
            (2, (5, 6, 9)),
        )
        for item_id in item_ids
    )
    return SampleData(num_users=3, items=items, interactions=interactions)
```

Create an empty `app/__init__.py`.

- [ ] **Step 4: Run the data tests and verify they pass**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_data.py' -v`

Expected: 3 tests pass.

- [ ] **Step 5: Commit the data foundation**

```bash
git add ml_sample/dpp-reranking/app ml_sample/dpp-reranking/tests/test_data.py
git commit -m "feat: add DPP sample data configuration"
```

### Task 2: User And Item Towers

**Files:**
- Create: `ml_sample/dpp-reranking/app/user_tower.py`
- Create: `ml_sample/dpp-reranking/app/item_tower.py`
- Create: `ml_sample/dpp-reranking/tests/test_towers.py`

- [ ] **Step 1: Write failing tests for both normalized towers**

```python
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.item_tower import ItemTower
from app.user_tower import UserTower


class TowerTests(unittest.TestCase):
    def test_towers_return_unit_normalized_embeddings(self) -> None:
        ids = torch.tensor([0, 2])
        for tower in (UserTower(3, 4), ItemTower(3, 4)):
            with self.subTest(tower=type(tower).__name__):
                output = tower(ids)
                self.assertEqual(output.shape, (2, 4))
                self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2), atol=1e-6))

    def test_towers_reject_non_positive_dimensions(self) -> None:
        for tower_type in (UserTower, ItemTower):
            with self.subTest(tower=tower_type.__name__), self.assertRaises(ValueError):
                tower_type(0, 4)
            with self.subTest(tower=tower_type.__name__), self.assertRaises(ValueError):
                tower_type(3, 0)
```

- [ ] **Step 2: Run the test and verify the missing-module failure**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_towers.py' -v`

Expected: FAIL because `app.item_tower` does not exist.

- [ ] **Step 3: Implement the two tower classes in separate files**

```python
# app/user_tower.py
import torch
from torch import nn
from torch.nn import functional


class UserTower(nn.Module):
    def __init__(self, num_users: int, embedding_dim: int) -> None:
        super().__init__()
        if num_users <= 0 or embedding_dim <= 0:
            raise ValueError("num_users and embedding_dim must be positive")
        self.embedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.embedding(user_ids), dim=-1)
```

```python
# app/item_tower.py
import torch
from torch import nn
from torch.nn import functional


class ItemTower(nn.Module):
    def __init__(self, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        if num_items <= 0 or embedding_dim <= 0:
            raise ValueError("num_items and embedding_dim must be positive")
        self.embedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_ids: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.embedding(item_ids), dim=-1)
```

- [ ] **Step 4: Run the tower tests and verify they pass**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_towers.py' -v`

Expected: 2 tests pass.

- [ ] **Step 5: Commit the separate towers**

```bash
git add ml_sample/dpp-reranking/app/user_tower.py ml_sample/dpp-reranking/app/item_tower.py ml_sample/dpp-reranking/tests/test_towers.py
git commit -m "feat: add user and item towers"
```

### Task 3: Two-Tower Model

**Files:**
- Create: `ml_sample/dpp-reranking/app/model.py`
- Create: `ml_sample/dpp-reranking/tests/test_model.py`

- [ ] **Step 1: Write the failing composition and gradient test**

```python
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model import TwoTowerModel


class ModelTests(unittest.TestCase):
    def test_scores_pairs_and_updates_both_towers(self) -> None:
        model = TwoTowerModel(num_users=2, num_items=3, embedding_dim=4)
        scores = model(torch.tensor([0, 1]), torch.tensor([1, 2]))

        self.assertEqual(scores.shape, (2,))
        scores.sum().backward()
        self.assertIsNotNone(model.user_tower.embedding.weight.grad)
        self.assertIsNotNone(model.item_tower.embedding.weight.grad)
```

- [ ] **Step 2: Run the test and verify the missing-model failure**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_model.py' -v`

Expected: FAIL because `app.model` does not exist.

- [ ] **Step 3: Implement tower composition and dot-product scoring**

```python
import torch
from torch import nn

from app.item_tower import ItemTower
from app.user_tower import UserTower


class TwoTowerModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_tower = UserTower(num_users, embedding_dim)
        self.item_tower = ItemTower(num_items, embedding_dim)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        users = self.user_tower(user_ids)
        items = self.item_tower(item_ids)
        return (users * items).sum(dim=-1)
```

- [ ] **Step 4: Run the model tests and verify they pass**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_model.py' -v`

Expected: 1 test passes.

- [ ] **Step 5: Commit the two-tower model**

```bash
git add ml_sample/dpp-reranking/app/model.py ml_sample/dpp-reranking/tests/test_model.py
git commit -m "feat: compose two tower recommender"
```

### Task 4: BPR Training

**Files:**
- Create: `ml_sample/dpp-reranking/app/train.py`
- Create: `ml_sample/dpp-reranking/tests/test_train.py`

- [ ] **Step 1: Write failing tests for learning and impossible negatives**

```python
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import TrainConfig
from app.data import Interaction
from app.model import TwoTowerModel
from app.train import train


class TrainTests(unittest.TestCase):
    def test_bpr_training_lowers_loss_and_ranks_positive_higher(self) -> None:
        torch.manual_seed(0)
        model = TwoTowerModel(1, 2, 4)
        losses = train(
            model,
            (Interaction(0, 0),),
            num_items=2,
            config=TrainConfig(embedding_dim=4, epochs=30, learning_rate=0.05, seed=3),
        )

        self.assertLess(losses[-1], losses[0])
        with torch.no_grad():
            scores = model(torch.tensor([0, 0]), torch.tensor([0, 1]))
        self.assertGreater(scores[0].item(), scores[1].item())

    def test_training_rejects_user_with_no_negative_item(self) -> None:
        model = TwoTowerModel(1, 1, 2)
        with self.assertRaisesRegex(ValueError, "negative item"):
            train(model, (Interaction(0, 0),), 1, TrainConfig(embedding_dim=2))
```

- [ ] **Step 2: Run the test and verify the missing-training failure**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_train.py' -v`

Expected: FAIL because `app.train` does not exist.

- [ ] **Step 3: Implement deterministic negative sampling and BPR optimization**

```python
import random
from collections import defaultdict
from collections.abc import Sequence

import torch
from torch.nn import functional

from app.config import TrainConfig
from app.data import Interaction
from app.model import TwoTowerModel


def train(
    model: TwoTowerModel,
    interactions: Sequence[Interaction],
    num_items: int,
    config: TrainConfig,
) -> list[float]:
    if not interactions:
        raise ValueError("interactions must not be empty")
    positives: dict[int, set[int]] = defaultdict(set)
    for interaction in interactions:
        positives[interaction.user_id].add(interaction.item_id)
    negatives = {
        user_id: tuple(item_id for item_id in range(num_items) if item_id not in item_ids)
        for user_id, item_ids in positives.items()
    }
    if any(not item_ids for item_ids in negatives.values()):
        raise ValueError("each user must have at least one negative item")

    torch.manual_seed(config.seed)
    random_generator = random.Random(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    epoch_losses: list[float] = []
    rows = list(interactions)

    for _ in range(config.epochs):
        random_generator.shuffle(rows)
        total_loss = 0.0
        for row in rows:
            negative_item_id = random_generator.choice(negatives[row.user_id])
            user_ids = torch.tensor([row.user_id])
            positive_ids = torch.tensor([row.item_id])
            negative_ids = torch.tensor([negative_item_id])
            loss = -functional.logsigmoid(
                model(user_ids, positive_ids) - model(user_ids, negative_ids)
            ).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        epoch_losses.append(total_loss / len(rows))
    return epoch_losses
```

- [ ] **Step 4: Run the training tests and verify they pass**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_train.py' -v`

Expected: 2 tests pass.

- [ ] **Step 5: Commit BPR training**

```bash
git add ml_sample/dpp-reranking/app/train.py ml_sample/dpp-reranking/tests/test_train.py
git commit -m "feat: train two tower model with BPR"
```

### Task 5: DPP Kernel Builder

**Files:**
- Create: `ml_sample/dpp-reranking/app/reranker.py`
- Create: `ml_sample/dpp-reranking/tests/test_reranker.py`

- [ ] **Step 1: Write failing kernel quality and validation tests**

```python
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.reranker import KernelBuilder


class KernelBuilderTests(unittest.TestCase):
    def test_builds_symmetric_positive_semidefinite_quality_kernel(self) -> None:
        kernel = KernelBuilder().build(
            torch.tensor([1.0, 0.0, -1.0]),
            torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
        )

        self.assertTrue(torch.isfinite(kernel).all())
        self.assertTrue(torch.allclose(kernel, kernel.T, atol=1e-6))
        self.assertGreaterEqual(torch.linalg.eigvalsh(kernel).min().item(), -1e-6)
        self.assertGreater(kernel[0, 0].item(), kernel[2, 2].item())

    def test_rejects_mismatched_candidate_counts(self) -> None:
        with self.assertRaises(ValueError):
            KernelBuilder().build(torch.ones(2), torch.ones(3, 2))

    def test_rejects_invalid_ranks_and_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            KernelBuilder().build(torch.ones(2, 1), torch.ones(2, 2))
        with self.assertRaises(ValueError):
            KernelBuilder().build(torch.tensor([1.0, float("nan")]), torch.ones(2, 2))

    def test_rejects_invalid_quality_scale(self) -> None:
        with self.assertRaises(ValueError):
            KernelBuilder(quality_scale=-1.0)
```

- [ ] **Step 2: Run the test and verify `KernelBuilder` is missing**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_reranker.py' -v`

Expected: FAIL because `app.reranker` does not exist or lacks `KernelBuilder`.

- [ ] **Step 3: Implement quality-weighted cosine kernel construction**

```python
import math

import torch
from torch.nn import functional


class KernelBuilder:
    def __init__(self, quality_scale: float = 1.0, epsilon: float = 1e-6) -> None:
        if not math.isfinite(quality_scale) or quality_scale < 0:
            raise ValueError("quality_scale must be finite and non-negative")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        self.quality_scale = quality_scale
        self.epsilon = epsilon

    def build(self, scores: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 1 or item_embeddings.ndim != 2:
            raise ValueError("scores and item_embeddings must have ranks 1 and 2")
        if scores.shape[0] != item_embeddings.shape[0]:
            raise ValueError("candidate counts must match")
        if not torch.isfinite(scores).all() or not torch.isfinite(item_embeddings).all():
            raise ValueError("scores and item_embeddings must be finite")
        normalized = functional.normalize(item_embeddings, dim=1)
        similarity = normalized @ normalized.T
        quality = torch.exp(self.quality_scale * scores)
        kernel = quality[:, None] * similarity * quality[None, :]
        identity = torch.eye(kernel.shape[0], dtype=kernel.dtype, device=kernel.device)
        return kernel + self.epsilon * identity
```

- [ ] **Step 4: Run the kernel tests and verify they pass**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_reranker.py' -v`

Expected: 4 tests pass.

- [ ] **Step 5: Commit kernel construction**

```bash
git add ml_sample/dpp-reranking/app/reranker.py ml_sample/dpp-reranking/tests/test_reranker.py
git commit -m "feat: build DPP quality kernel"
```

### Task 6: Fast Greedy MAP Selector And Reranker

**Files:**
- Modify: `ml_sample/dpp-reranking/app/reranker.py`
- Modify: `ml_sample/dpp-reranking/tests/test_reranker.py`

- [ ] **Step 1: Add failing selector and facade tests**

Append these imports and test classes to `tests/test_reranker.py`:

```python
from app.reranker import DPPReranker, GreedyMapSelector


class GreedyMapSelectorTests(unittest.TestCase):
    def test_selects_quality_then_complementary_candidate(self) -> None:
        kernel = torch.tensor(
            [[4.0, 3.9, 0.0], [3.9, 4.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertEqual(GreedyMapSelector().select(kernel, 2), [0, 2])

    def test_ties_follow_original_candidate_order(self) -> None:
        self.assertEqual(GreedyMapSelector().select(torch.eye(3), 2), [0, 1])

    def test_rejects_invalid_selection_count(self) -> None:
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.eye(2), 3)

    def test_rejects_non_square_and_non_finite_kernels(self) -> None:
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.ones(2, 3), 1)
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.tensor([[float("nan")]]), 1)


class DPPRerankerTests(unittest.TestCase):
    def test_reranks_candidates_through_builder_and_selector(self) -> None:
        indices = DPPReranker().rerank(
            torch.tensor([1.0, 0.9, 0.1]),
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            top_k=2,
        )
        self.assertEqual(indices, [0, 2])
```

- [ ] **Step 2: Run the reranker tests and verify the missing-class failure**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_reranker.py' -v`

Expected: FAIL because `GreedyMapSelector` and `DPPReranker` are not defined.

- [ ] **Step 3: Implement incremental-Cholesky Fast Greedy MAP and composition**

Append to `app/reranker.py`:

```python
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
```

- [ ] **Step 4: Run all reranker tests and verify they pass**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_reranker.py' -v`

Expected: 9 tests pass.

- [ ] **Step 5: Commit Fast Greedy MAP reranking**

```bash
git add ml_sample/dpp-reranking/app/reranker.py ml_sample/dpp-reranking/tests/test_reranker.py
git commit -m "feat: add fast greedy MAP reranker"
```

### Task 7: Recommendation Display

**Files:**
- Create: `ml_sample/dpp-reranking/app/utils.py`
- Create: `ml_sample/dpp-reranking/tests/test_utils.py`

- [ ] **Step 1: Write the failing output-format test**

```python
import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data import Recommendation
from app.utils import show


class UtilsTests(unittest.TestCase):
    def test_show_formats_both_lists_to_supplied_stream(self) -> None:
        output = StringIO()
        show(
            [Recommendation(1, "books", 0.75)],
            [Recommendation(2, "music", 0.50)],
            stream=output,
        )

        text = output.getvalue()
        self.assertIn("Relevance ranking", text)
        self.assertIn("DPP ranking", text)
        self.assertIn("books", text)
        self.assertIn("music", text)
        self.assertIn("0.7500", text)
```

- [ ] **Step 2: Run the test and verify the missing-utility failure**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_utils.py' -v`

Expected: FAIL because `app.utils` does not exist.

- [ ] **Step 3: Implement `show` as the only presentation owner**

```python
import sys
from collections.abc import Sequence
from typing import TextIO

from app.data import Recommendation


def show(
    ordinary: Sequence[Recommendation],
    reranked: Sequence[Recommendation],
    stream: TextIO | None = None,
) -> None:
    output = sys.stdout if stream is None else stream
    for title, recommendations in (
        ("Relevance ranking", ordinary),
        ("DPP ranking", reranked),
    ):
        print(title, file=output)
        print("rank  item  category  score", file=output)
        for rank, recommendation in enumerate(recommendations, start=1):
            print(
                f"{rank:>4}  {recommendation.item_id:>4}  "
                f"{recommendation.category:<8}  {recommendation.score:.4f}",
                file=output,
            )
        print(file=output)
```

- [ ] **Step 4: Run the utility test and verify it passes**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_utils.py' -v`

Expected: 1 test passes.

- [ ] **Step 5: Commit display handling**

```bash
git add ml_sample/dpp-reranking/app/utils.py ml_sample/dpp-reranking/tests/test_utils.py
git commit -m "feat: display recommendation comparisons"
```

### Task 8: End-To-End CLI

**Files:**
- Create: `ml_sample/dpp-reranking/app/main.py`
- Create: `ml_sample/dpp-reranking/tests/test_main.py`

- [ ] **Step 1: Write the failing end-to-end output test**

```python
import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import main


class MainTests(unittest.TestCase):
    def test_main_runs_training_and_prints_both_rankings(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            main()

        text = output.getvalue()
        self.assertIn("Relevance ranking", text)
        self.assertIn("DPP ranking", text)
        self.assertEqual(text.count("rank  item  category  score"), 2)
```

- [ ] **Step 2: Run the test and verify the missing-main failure**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_main.py' -v`

Expected: FAIL because `app.main` does not exist.

- [ ] **Step 3: Implement orchestration without formatting or DPP internals**

```python
import torch

from app.config import TrainConfig
from app.data import Recommendation, create_sample_data
from app.model import TwoTowerModel
from app.reranker import DPPReranker
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
```

- [ ] **Step 4: Run the main test and then the demo**

Run: `cd ml_sample/dpp-reranking && python -m unittest discover -s tests -p 'test_main.py' -v`

Expected: 1 test passes.

Run: `cd ml_sample/dpp-reranking && python -m app.main`

Expected: two five-row sections headed `Relevance ranking` and `DPP ranking`, with item IDs, categories, and finite scores.

- [ ] **Step 5: Commit the end-to-end flow**

```bash
git add ml_sample/dpp-reranking/app/main.py ml_sample/dpp-reranking/tests/test_main.py
git commit -m "feat: run DPP reranking demo"
```

### Task 9: Project Entry Points And Documentation

**Files:**
- Modify: `ml_sample/dpp-reranking/README.md`
- Create: `ml_sample/dpp-reranking/Makefile`
- Create: `ml_sample/dpp-reranking/requirements.txt`

- [ ] **Step 1: Add dependency and Make entry points**

```text
# requirements.txt
torch>=2.2,<3
```

```makefile
PYTHON ?= python

.PHONY: run test

run:
	$(PYTHON) -m app.main

test:
	$(PYTHON) -m unittest discover -s tests -v
```

- [ ] **Step 2: Document setup, architecture, equations, and commands**

Write `README.md` with these concrete sections:

````markdown
# DPP Reranking

This sample trains separate user and item embedding towers with BPR loss, ranks
unseen items by dot-product relevance, and uses DPP Fast Greedy MAP to select a
relevant but less redundant top-k list.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
make run
make test
```

## Model And Reranker

`UserTower` and `ItemTower` produce normalized embeddings. `TwoTowerModel`
scores a pair by their dot product and `train` optimizes BPR loss:

$$
-\log \sigma(s(u,i^+) - s(u,i^-)).
$$

`KernelBuilder` combines quality $q_i = \exp(\alpha r_i)$ and learned cosine
similarity into $L_{ij}=q_i(\phi_i^\top\phi_j)q_j$. `GreedyMapSelector` uses
incremental Cholesky updates to perform Fast Greedy MAP in approximately
$O(Nk^2)$ time. `DPPReranker` composes both operations.

The in-memory data is intentionally small and deterministic. Categories make
the difference between the relevance-only and diversified result lists easy to
inspect; no external dataset is downloaded.
````

- [ ] **Step 3: Run the complete verification suite**

Run: `cd ml_sample/dpp-reranking && make test`

Expected: all tests pass with no errors or warnings.

Run: `cd ml_sample/dpp-reranking && make run`

Expected: relevance and DPP ranking tables are printed successfully.

- [ ] **Step 4: Check diagnostics and diff quality**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 5: Commit project entry points and documentation**

```bash
git add ml_sample/dpp-reranking/README.md ml_sample/dpp-reranking/Makefile ml_sample/dpp-reranking/requirements.txt
git commit -m "docs: explain DPP reranking sample"
```