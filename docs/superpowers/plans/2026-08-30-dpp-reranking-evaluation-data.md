# DPP Reranking Evaluation Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate model-training interactions from reranking evaluation interactions and print 20 results in each ranking.

**Architecture:** Two `SampleData` factories share a deterministic 30-item catalog and user ID space while returning distinct interaction sets. `main()` trains from one dataset and ranks candidates from the other without changing model, ranker, or DPP interfaces.

**Tech Stack:** Python 3.11, PyTorch, `unittest`

---

### Task 1: Specify separate datasets

**Files:**
- Modify: `ml_sample/dpp-reranking/tests/test_data.py`
- Modify: `ml_sample/dpp-reranking/app/data.py`

- [ ] Add a failing data test importing `create_training_data` and `create_reranking_data` and asserting both catalogs contain the same 30 items, interaction sets differ, and user 0 has 27 reranking candidates.
- [ ] Run `python -m unittest tests.test_data -v`; expect an import failure because the factories do not exist.
- [ ] Replace `create_sample_data` with the two factories backed by one private 30-item catalog factory and distinct valid interactions.
- [ ] Rerun `python -m unittest tests.test_data -v`; expect all data tests to pass.

### Task 2: Specify 20-row output

**Files:**
- Modify: `ml_sample/dpp-reranking/tests/test_main.py`
- Modify: `ml_sample/dpp-reranking/app/main.py`

- [ ] Strengthen the main integration test to split both output sections and assert each contains 20 recommendation rows.
- [ ] Run `python -m unittest tests.test_main -v`; expect failure because current output contains five rows per section.
- [ ] Update `main()` to train with `create_training_data()`, rank with `create_reranking_data()`, and set `top_k = 20`.
- [ ] Rerun `python -m unittest tests.test_main -v`; expect the integration test to pass.

### Task 3: Verify and commit

**Files:**
- Test: `ml_sample/dpp-reranking/tests/`

- [ ] Run `make test && make run`; expect all tests to pass and 20 rows under each output heading.
- [ ] Run editor diagnostics for changed Python files and `git diff --check`; expect no errors.
- [ ] Commit the plan and implementation with `git commit -m "feat: separate DPP evaluation data"`.
