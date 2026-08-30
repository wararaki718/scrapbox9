# DPP Model Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the two-tower model classes into `app.models` without changing behavior.

**Architecture:** Each existing model class keeps its own file under `app/models/`. The package `__init__.py` re-exports all three classes, and all application and test consumers use that package API.

**Tech Stack:** Python 3.11, PyTorch, unittest

---

### Task 1: Move Model Classes Into The Models Package

**Files:**
- Create: `ml_sample/dpp-reranking/app/models/__init__.py`
- Create: `ml_sample/dpp-reranking/app/models/model.py`
- Create: `ml_sample/dpp-reranking/app/models/user_tower.py`
- Create: `ml_sample/dpp-reranking/app/models/item_tower.py`
- Delete: `ml_sample/dpp-reranking/app/model.py`
- Delete: `ml_sample/dpp-reranking/app/user_tower.py`
- Delete: `ml_sample/dpp-reranking/app/item_tower.py`
- Modify: `ml_sample/dpp-reranking/app/main.py`
- Modify: `ml_sample/dpp-reranking/app/train.py`
- Modify: `ml_sample/dpp-reranking/tests/test_model.py`
- Modify: `ml_sample/dpp-reranking/tests/test_towers.py`
- Modify: `ml_sample/dpp-reranking/tests/test_train.py`

- [ ] **Step 1: Change tests to the new package API**

Use these imports where each class is needed:

```python
from app.models import ItemTower, TwoTowerModel, UserTower
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `cd ml_sample/dpp-reranking && python -m unittest tests.test_model tests.test_towers tests.test_train -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'app.models'`.

- [ ] **Step 3: Move the classes without changing their behavior**

Place the existing `UserTower` in `app/models/user_tower.py`, the existing
`ItemTower` in `app/models/item_tower.py`, and the existing `TwoTowerModel` in
`app/models/model.py`. In `model.py`, import both towers through the package API:

```python
from app.models.item_tower import ItemTower
from app.models.user_tower import UserTower
```

Define the package API in `app/models/__init__.py`:

```python
from app.models.item_tower import ItemTower
from app.models.model import TwoTowerModel
from app.models.user_tower import UserTower

__all__ = ["ItemTower", "TwoTowerModel", "UserTower"]
```

Delete the three original modules after their contents have moved.

- [ ] **Step 4: Move application imports to `app.models`**

Use this import in `app/main.py` and `app/train.py`:

```python
from app.models import TwoTowerModel
```

- [ ] **Step 5: Run focused and complete verification**

Run: `cd ml_sample/dpp-reranking && python -m unittest tests.test_model tests.test_towers tests.test_train -v`

Expected: 5 tests pass.

Run: `cd ml_sample/dpp-reranking && make test && make run`

Expected: all 19 tests pass, followed by relevance and DPP ranking tables.

- [ ] **Step 6: Commit the refactor**

```bash
git add -A ml_sample/dpp-reranking/app ml_sample/dpp-reranking/tests
git commit -m "refactor: move DPP models into package"
```