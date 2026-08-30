# DPP Reranking Evaluate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare relevance reranking and DPP with deterministic dummy judgments using NDCG, intra-list diversity, and category coverage.

**Architecture:** Independent metric functions live under `app/evaluate/metrics`. `Evaluator` selects 20 items from the full candidate pool with DPP, restores relevance-score order inside that subset, computes both ranking rows, and returns immutable results for `main()` to print.

**Tech Stack:** Python 3.11, PyTorch, standard-library `unittest`

---

### Task 1: Implement NDCG@k

**Files:**
- Create: `ml_sample/dpp-reranking/app/evaluate/__init__.py`
- Create: `ml_sample/dpp-reranking/app/evaluate/metrics/__init__.py`
- Create: `ml_sample/dpp-reranking/app/evaluate/metrics/ndcg.py`
- Create: `ml_sample/dpp-reranking/tests/evaluate/__init__.py`
- Create: `ml_sample/dpp-reranking/tests/evaluate/metrics/__init__.py`
- Create: `ml_sample/dpp-reranking/tests/evaluate/metrics/test_ndcg.py`

- [ ] Write tests for perfect ranking equal to 1, reversed ranking below 1, missing judgments as zero, zero ideal DCG, and non-positive cutoff rejection.
- [ ] Run `python -m unittest tests.evaluate.metrics.test_ndcg -v`; expect import failure.
- [ ] Implement `ndcg_at_k(item_ids, relevance_by_item_id, k)` with graded gain $2^{rel}-1$ and logarithmic discount.
- [ ] Rerun the focused test; expect pass.

### Task 2: Implement diversity metrics

**Files:**
- Create: `ml_sample/dpp-reranking/app/evaluate/metrics/intra_list_diversity.py`
- Create: `ml_sample/dpp-reranking/app/evaluate/metrics/category_coverage.py`
- Create: `ml_sample/dpp-reranking/tests/evaluate/metrics/test_intra_list_diversity.py`
- Create: `ml_sample/dpp-reranking/tests/evaluate/metrics/test_category_coverage.py`

- [ ] Write ILD tests for orthogonal vectors, a single item, missing/non-finite/mismatched embeddings, and invalid cutoff.
- [ ] Write category coverage tests for distinct category count, cutoff behavior, missing metadata, and invalid cutoff.
- [ ] Run both focused modules; expect import failures.
- [ ] Implement mean pairwise cosine distance in `intra_list_diversity_at_k` and distinct category count in `category_coverage_at_k`.
- [ ] Rerun both focused modules; expect pass.

### Task 3: Implement Evaluator

**Files:**
- Create: `ml_sample/dpp-reranking/app/evaluate/evaluator.py`
- Modify: `ml_sample/dpp-reranking/app/evaluate/__init__.py`
- Create: `ml_sample/dpp-reranking/tests/evaluate/test_evaluator.py`

- [ ] Write a test with a deterministic `CandidateRanking`, dummy judgments, and injected DPP reranker; verify DPP receives the full pool, selects 20, selected items are restored to relevance order, and result rows contain NDCG@5/10/20, ILD@20, and coverage@20.
- [ ] Run `python -m unittest tests.evaluate.test_evaluator -v`; expect import failure.
- [ ] Implement immutable `EvaluationResult` and `Evaluator.evaluate(ranking, relevance_by_item_id, selection_count=20, cutoffs=(5, 10, 20))`.
- [ ] Rerun the focused test; expect pass.

### Task 4: Integrate output and documentation

**Files:**
- Modify: `ml_sample/dpp-reranking/app/main.py`
- Modify: `ml_sample/dpp-reranking/app/utils.py`
- Modify: `ml_sample/dpp-reranking/tests/test_main.py`
- Modify: `ml_sample/dpp-reranking/tests/test_utils.py`
- Modify: `ml_sample/dpp-reranking/README.md`

- [ ] Add output tests requiring an evaluation table with rows for `Reranking 1` and `DPP` and NDCG@5/10/20, ILD@20, coverage@20 columns.
- [ ] Run `python -m unittest tests.test_utils tests.test_main -v`; expect failure because no evaluation output exists.
- [ ] Add `show_evaluation`, build deterministic category-based graded dummy judgments in `main()`, evaluate the all-candidate ranking with selection count 20, and print after recommendation tables.
- [ ] Document $M$, $K$, and $N$, the fixed-setting rule $M>K\ge\max(N)$, relevance reordering, invalid $M=K=N$, and complementary diversity metrics in README.
- [ ] Rerun focused integration tests; expect pass.

### Task 5: Verify and commit

**Files:**
- Test: `ml_sample/dpp-reranking/tests/`

- [ ] Run `make test && make run`; expect all tests to pass and evaluation rows to print.
- [ ] Run diagnostics for changed Python files and `git diff --check`; expect no errors.
- [ ] Commit with `git commit -m "feat: evaluate DPP reranking"`.
