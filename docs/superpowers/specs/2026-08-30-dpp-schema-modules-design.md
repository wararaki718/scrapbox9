# DPP Schema Modules Design

## Goal

Move each DPP reranking dataclass into its own module under
`ml_sample/dpp-reranking/app/schemas/` without changing runtime behavior.

## Structure

- `app/schemas/item.py` defines `Item`.
- `app/schemas/interaction.py` defines `Interaction`.
- `app/schemas/recommendation.py` defines `Recommendation`.
- `app/schemas/sample_data.py` defines `SampleData` and imports `Item` and
  `Interaction` from their schema modules.
- `app/schemas/__init__.py` re-exports all four classes as the package API.
- `app/data.py` retains only `create_sample_data()` and imports the schema
  classes from `app.schemas`.

Application modules and tests import dataclasses from `app.schemas`. There is
no compatibility re-export from `app.data`, because this is a new sample with
no external API compatibility requirement.

## Validation

Tests first adopt the new `app.schemas` import contract and must fail while the
package is absent. After moving the classes, all existing tests must pass and
`make run` must produce the same relevance and DPP ranking tables.