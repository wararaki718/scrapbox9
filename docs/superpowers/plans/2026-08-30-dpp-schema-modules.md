# DPP Schema Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move every DPP sample dataclass into its own `app.schemas` module without changing behavior.

**Architecture:** Each schema module owns one frozen dataclass. `app.schemas.__init__` re-exports the four classes, while `app.data` retains only sample-data construction and all consumers import schemas through the package API.

**Tech Stack:** Python 3.11, dataclasses, unittest

---

### Task 1: Split Dataclasses Into Schema Modules

**Files:**
- Create: `ml_sample/dpp-reranking/app/schemas/__init__.py`
- Create: `ml_sample/dpp-reranking/app/schemas/item.py`
- Create: `ml_sample/dpp-reranking/app/schemas/interaction.py`
- Create: `ml_sample/dpp-reranking/app/schemas/recommendation.py`
- Create: `ml_sample/dpp-reranking/app/schemas/sample_data.py`
- Modify: `ml_sample/dpp-reranking/app/data.py`
- Modify: `ml_sample/dpp-reranking/app/main.py`
- Modify: `ml_sample/dpp-reranking/app/train.py`
- Modify: `ml_sample/dpp-reranking/app/utils.py`
- Modify: `ml_sample/dpp-reranking/tests/test_data.py`
- Modify: `ml_sample/dpp-reranking/tests/test_train.py`
- Modify: `ml_sample/dpp-reranking/tests/test_utils.py`

- [ ] **Step 1: Change tests to the new schema package API**

Replace dataclass imports in tests with:

```python
from app.schemas import Interaction, Recommendation
```

Each test imports only the names it uses. `create_sample_data` remains imported
from `app.data`.

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `cd ml_sample/dpp-reranking && python -m unittest tests.test_data tests.test_train tests.test_utils -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.schemas'`.

- [ ] **Step 3: Create one frozen dataclass per schema module**

```python
# app/schemas/item.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    item_id: int
    category: str
```

```python
# app/schemas/interaction.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Interaction:
    user_id: int
    item_id: int
```

```python
# app/schemas/recommendation.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Recommendation:
    item_id: int
    category: str
    score: float
```

```python
# app/schemas/sample_data.py
from dataclasses import dataclass

from app.schemas.interaction import Interaction
from app.schemas.item import Item


@dataclass(frozen=True)
class SampleData:
    num_users: int
    items: tuple[Item, ...]
    interactions: tuple[Interaction, ...]
```

```python
# app/schemas/__init__.py
from app.schemas.interaction import Interaction
from app.schemas.item import Item
from app.schemas.recommendation import Recommendation
from app.schemas.sample_data import SampleData

__all__ = ["Interaction", "Item", "Recommendation", "SampleData"]
```

- [ ] **Step 4: Move all application imports to `app.schemas`**

`app/data.py` imports `Interaction`, `Item`, and `SampleData`; `app/main.py` and
`app/utils.py` import `Recommendation`; `app/train.py` imports `Interaction`.
Remove all dataclass definitions and the `dataclasses` import from `app/data.py`.

- [ ] **Step 5: Run focused and complete verification**

Run: `cd ml_sample/dpp-reranking && python -m unittest tests.test_data tests.test_train tests.test_utils -v`

Expected: 6 tests pass.

Run: `cd ml_sample/dpp-reranking && make test && make run`

Expected: all 19 tests pass, followed by relevance and DPP ranking tables.

- [ ] **Step 6: Commit the refactor**

```bash
git add ml_sample/dpp-reranking/app ml_sample/dpp-reranking/tests
git commit -m "refactor: split DPP schemas into modules"
```