# DPP Reranking Test Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mirror the `app` module layout under `tests` while preserving every existing behavioral assertion.

**Architecture:** Keep top-level module tests at the test root and introduce `models`, `reranker`, and `schemas` test packages matching production packages. Split mixed test classes by the production class they exercise and remove only the brittle module-metadata assertion.

**Tech Stack:** Python 3.11, `unittest`, PyTorch

---

### Task 1: Separate top-level and schema tests

**Files:**
- Create: `ml_sample/dpp-reranking/tests/test_config.py`
- Modify: `ml_sample/dpp-reranking/tests/test_data.py`
- Create: `ml_sample/dpp-reranking/tests/schemas/__init__.py`
- Create: `ml_sample/dpp-reranking/tests/schemas/test_recommendation.py`

- [ ] **Step 1: Move config validation into `test_config.py`**

Create a `TrainConfigTests` class containing the existing invalid-value loop and import only `TrainConfig`.

- [ ] **Step 2: Move recommendation immutability into the schema package**

Create `RecommendationTests.test_is_immutable` with the existing assignment assertion. Set the project root using `Path(__file__).resolve().parents[2]` in the nested test.

- [ ] **Step 3: Keep only sample-data behavior in `test_data.py`**

Remove `TrainConfig`, `Recommendation`, and their tests, retaining `DataTests.test_sample_data_has_contiguous_ids_and_valid_interactions`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m unittest tests.test_config tests.test_data tests.schemas.test_recommendation -v
```

Expected: 3 tests pass.

### Task 2: Mirror model tests

**Files:**
- Create: `ml_sample/dpp-reranking/tests/models/__init__.py`
- Move: `ml_sample/dpp-reranking/tests/test_model.py` to `ml_sample/dpp-reranking/tests/models/test_model.py`
- Replace: `ml_sample/dpp-reranking/tests/test_towers.py` with `ml_sample/dpp-reranking/tests/models/test_user_tower.py`
- Create: `ml_sample/dpp-reranking/tests/models/test_item_tower.py`

- [ ] **Step 1: Move the two-tower model test**

Retain `test_scores_pairs_and_updates_both_towers`, changing the nested test root to `parents[2]`.

- [ ] **Step 2: Split tower tests by class**

Each tower test class must independently verify unit-normalized output shape and rejection of non-positive vocabulary and embedding dimensions. Use `UserTowerTests` and `ItemTowerTests` with direct imports from `app.models`.

- [ ] **Step 3: Run focused tests**

Run:

```bash
python -m unittest discover -s tests/models -v
```

Expected: 5 tests pass.

### Task 3: Mirror reranker tests

**Files:**
- Create: `ml_sample/dpp-reranking/tests/reranker/__init__.py`
- Replace: `ml_sample/dpp-reranking/tests/test_reranker.py` with `ml_sample/dpp-reranking/tests/reranker/test_builder.py`
- Create: `ml_sample/dpp-reranking/tests/reranker/test_selector.py`
- Create: `ml_sample/dpp-reranking/tests/reranker/test_reranker.py`

- [ ] **Step 1: Move all `KernelBuilderTests` assertions to `test_builder.py`**

Retain symmetric PSD, candidate count, invalid rank/non-finite input, and quality scale tests.

- [ ] **Step 2: Move all `GreedyMapSelectorTests` assertions to `test_selector.py`**

Retain complementary selection, stable ties, invalid count, and invalid kernel tests.

- [ ] **Step 3: Move `DPPRerankerTests` to `test_reranker.py`**

Retain the facade result assertion and remove `RerankerModuleTests`, which tests internal `__module__` metadata rather than behavior.

- [ ] **Step 4: Run focused tests**

Run:

```bash
python -m unittest discover -s tests/reranker -v
```

Expected: 9 tests pass.

### Task 4: Verify and commit

**Files:**
- Test: `ml_sample/dpp-reranking/tests/`

- [ ] **Step 1: Run the complete suite**

Run:

```bash
make test
```

Expected: 23 tests pass. The count increases from 22 because the shared two-tower assertions become four class-specific tests while the one metadata assertion is removed.

- [ ] **Step 2: Check diagnostics and whitespace**

Run editor diagnostics for all new test modules and `git diff --check`; expect no errors.

- [ ] **Step 3: Commit**

```bash
git add ml_sample/dpp-reranking/tests docs/superpowers/plans/2026-08-30-dpp-reranking-test-structure.md
git commit -m "test: align DPP tests with app structure"
```
