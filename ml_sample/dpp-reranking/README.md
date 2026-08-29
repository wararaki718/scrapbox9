# DPP Reranking

This sample trains separate user and item embedding towers with BPR loss, ranks
unseen items by dot-product relevance, and uses DPP Fast Greedy MAP to select a
relevant but less redundant top-k list.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
make run
make test
```

## Model And Reranker

`UserTower` and `ItemTower` produce normalized embeddings. `TwoTowerModel`
scores a pair by their dot product and `train` optimizes BPR loss:

$$
-\log \sigma(s(u,i^+) - s(u,i^-)).
$$

`KernelBuilder` combines quality $q_i = \exp(\alpha r_i)$ and learned cosine
similarity into $L_{ij}=q_i(\phi_i^\top\phi_j)q_j$. `GreedyMapSelector` uses
incremental Cholesky updates to perform Fast Greedy MAP in approximately
$O(Nk^2)$ time. `DPPReranker` composes both operations.

The in-memory data is intentionally small and deterministic. Categories make
the difference between the relevance-only and diversified result lists easy to
inspect; no external dataset is downloaded.
