# DPP Reranker Modules Design

## Goal

Move each DPP reranking class into its own module under
`ml_sample/dpp-reranking/app/reranker/` without changing numerical behavior.

## Structure

- `app/reranker/builder.py` defines `KernelBuilder`.
- `app/reranker/selector.py` defines `GreedyMapSelector`.
- `app/reranker/reranker.py` defines `DPPReranker` and composes the builder and
  selector.
- `app/reranker/__init__.py` re-exports all three classes as the package API.

Application modules and tests continue to import classes from `app.reranker`.
The old `app/reranker.py` module is removed. No compatibility wrapper is needed
because the package preserves the existing public import path.

## Validation

A structure test first requires each class to originate from its corresponding
package module and must fail while the implementation remains monolithic. After
the move, all existing numerical and validation tests must pass, and `make run`
must produce the same relevance and DPP ranking tables.