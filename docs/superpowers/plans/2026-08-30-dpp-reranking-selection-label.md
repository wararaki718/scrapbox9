# DPP Reranking Selection Label Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select 10 DPP recommendations from the displayed top-20 relevance pool and label selected rows.

**Architecture:** `main()` slices aligned recommendations, scores, and embeddings to 20 before invoking DPP with a selection count of 10. `show()` derives selected item IDs from the reranked recommendations and renders a label column without changing recommendation schemas.

**Tech Stack:** Python 3.11, PyTorch, `unittest`

---

### Task 1: Label selected recommendations

**Files:**
- Modify: `ml_sample/dpp-reranking/tests/test_utils.py`
- Modify: `ml_sample/dpp-reranking/app/utils.py`

- [ ] Extend the output test with two relevance recommendations where only one appears in the DPP list; assert selected rows contain `reranked`, unselected rows contain `-`, and the header includes `label`.
- [ ] Run `python -m unittest tests.test_utils -v`; expect failure because no label column exists.
- [ ] Update `show()` to derive selected item IDs and append `reranked` or `-` to each row.
- [ ] Rerun `python -m unittest tests.test_utils -v`; expect the test to pass.

### Task 2: Select 10 from the top 20

**Files:**
- Modify: `ml_sample/dpp-reranking/tests/test_main.py`
- Modify: `ml_sample/dpp-reranking/app/main.py`

- [ ] Update the integration test to expect 20 relevance rows, 10 DPP rows, 10 `reranked` labels in each section, and DPP item IDs that are a subset of relevance item IDs.
- [ ] Run `python -m unittest tests.test_main -v`; expect failure because DPP currently outputs 20 rows.
- [ ] Introduce `candidate_count = 20` and `rerank_count = 10`; slice scores and embeddings to the candidate count before reranking and display only the same 20 recommendations.
- [ ] Rerun `python -m unittest tests.test_main -v`; expect the test to pass.

### Task 3: Verify and commit

**Files:**
- Test: `ml_sample/dpp-reranking/tests/`

- [ ] Run `make test && make run`; expect all tests to pass and the CLI to show 20 relevance rows and 10 DPP rows with labels.
- [ ] Run diagnostics for changed Python files and `git diff --check`; expect no errors.
- [ ] Commit with `git commit -m "feat: label DPP-selected recommendations"`.
