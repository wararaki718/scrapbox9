# DPP Reranking Test Structure Design

## Goal

Align the test layout with the current `app` package structure so each production module has an obvious corresponding test module. Preserve all existing behavior and coverage.

## Structure

Keep tests for top-level application modules directly under `tests/`:

- `test_config.py`
- `test_data.py`
- `test_main.py`
- `test_ranker.py`
- `test_train.py`
- `test_utils.py`

Mirror production subpackages for tests that already exercise those modules:

```text
tests/
  models/
    __init__.py
    test_model.py
    test_user_tower.py
    test_item_tower.py
  reranker/
    __init__.py
    test_builder.py
    test_selector.py
    test_reranker.py
  schemas/
    __init__.py
    test_recommendation.py
```

Do not create empty test files for production modules without dedicated behavior tests.

## Test Separation

- Move `TrainConfig` validation from `test_data.py` to `test_config.py`.
- Move `Recommendation` immutability from `test_data.py` to `schemas/test_recommendation.py`.
- Split the shared tower tests into equivalent `UserTower` and `ItemTower` tests.
- Split kernel builder, greedy selector, and DPP facade tests into their matching reranker modules.
- Remove the test that asserts implementation `__module__` strings. The mirrored layout makes ownership explicit, while behavior tests should not lock internal module metadata.

## Compatibility

Add `__init__.py` to nested test directories so standard-library `unittest` discovery recurses into them reliably. Nested tests will calculate the project root from their deeper path so they remain runnable both through `make test` and as individual modules.

## Validation

Run the focused tests for each new package, then run `make test`. The resulting suite must retain the existing 22 behavior checks except for replacing the module-metadata assertion with the separated tower coverage, leaving the behavior count unchanged or higher.
