# DPP Reranker Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the DPP reranking classes into focused modules under `app.reranker` without changing behavior.

**Architecture:** Kernel construction, Fast Greedy MAP selection, and orchestration each receive one module. The package `__init__.py` preserves the existing `from app.reranker import ...` API.

**Tech Stack:** Python 3.11, PyTorch, unittest

---

### Task 1: Split Reranker Classes Into Package Modules

**Files:**
- Create: `ml_sample/dpp-reranking/app/reranker/__init__.py`
- Create: `ml_sample/dpp-reranking/app/reranker/builder.py`
- Create: `ml_sample/dpp-reranking/app/reranker/selector.py`
- Create: `ml_sample/dpp-reranking/app/reranker/reranker.py`
- Delete: `ml_sample/dpp-reranking/app/reranker.py`
- Modify: `ml_sample/dpp-reranking/tests/test_reranker.py`

- [ ] **Step 1: Add a failing module-ownership test**

Add this test to `tests/test_reranker.py`:

```python
class RerankerModuleTests(unittest.TestCase):
    def test_each_class_is_defined_in_its_own_module(self) -> None:
        self.assertEqual(KernelBuilder.__module__, "app.reranker.builder")
        self.assertEqual(GreedyMapSelector.__module__, "app.reranker.selector")
        self.assertEqual(DPPReranker.__module__, "app.reranker.reranker")
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `cd ml_sample/dpp-reranking && python -m unittest tests.test_reranker.RerankerModuleTests -v`

Expected: FAIL because all classes currently report `app.reranker`.

- [ ] **Step 3: Move each class into its module**

Move `KernelBuilder` and its `math`, `torch`, and `functional` imports to
`app/reranker/builder.py`. Move `GreedyMapSelector` and its `math` and `torch`
imports to `app/reranker/selector.py`. Move `DPPReranker` to
`app/reranker/reranker.py` with these imports:

```python
import torch

from app.reranker.builder import KernelBuilder
from app.reranker.selector import GreedyMapSelector
```

Define the package API in `app/reranker/__init__.py`:

```python
from app.reranker.builder import KernelBuilder
from app.reranker.reranker import DPPReranker
from app.reranker.selector import GreedyMapSelector

__all__ = ["DPPReranker", "GreedyMapSelector", "KernelBuilder"]
```

Delete the original `app/reranker.py` after all class definitions have moved.

- [ ] **Step 4: Run focused and complete verification**

Run: `cd ml_sample/dpp-reranking && python -m unittest tests.test_reranker -v`

Expected: all 10 reranker tests pass.

Run: `cd ml_sample/dpp-reranking && make test && make run`

Expected: all 20 tests pass, followed by relevance and DPP ranking tables.

- [ ] **Step 5: Commit the refactor**

```bash
git add -A ml_sample/dpp-reranking/app ml_sample/dpp-reranking/tests/test_reranker.py
git commit -m "refactor: split DPP reranker into modules"
```