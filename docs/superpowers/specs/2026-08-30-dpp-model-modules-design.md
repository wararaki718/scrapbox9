# DPP Model Modules Design

## Goal

Move the DPP reranking model classes under
`ml_sample/dpp-reranking/app/models/` without changing model behavior.

## Structure

- `app/models/user_tower.py` defines `UserTower`.
- `app/models/item_tower.py` defines `ItemTower`.
- `app/models/model.py` defines `TwoTowerModel` and composes both towers.
- `app/models/__init__.py` re-exports all three classes as the package API.

Application modules and tests import model classes from `app.models`. The old
`app/model.py`, `app/user_tower.py`, and `app/item_tower.py` files are removed;
no compatibility wrappers are retained because this sample has no external API
compatibility requirement.

## Validation

Tests first adopt the new `app.models` import contract and must fail while the
package is absent. After moving the classes, all existing tests must pass and
`make run` must produce the same relevance and DPP ranking tables.