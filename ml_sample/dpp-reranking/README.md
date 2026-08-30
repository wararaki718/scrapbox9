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

## Offline Evaluation

`make run` also compares Reranking 1 with DPP using deterministic dummy graded
relevance judgments. The evaluation reports NDCG@5, NDCG@10, NDCG@20,
intra-list diversity (ILD)@20, and category coverage@20.

Three sizes must be configured independently:

- $M$: candidates passed from Reranking 1 to DPP.
- $K$: items selected by DPP.
- $N$: cutoff used by NDCG@$N$.

For one production ranking evaluated at several cutoffs, keep the DPP setting
fixed and choose

$$
M > K \ge \max(N).
$$

For example, NDCG@5, @10, and @20 can compare a Reranking 1 list with 20 items
selected by DPP from 100 candidates. This sample uses all $M=27$ unseen dummy
candidates and selects $K=20$ because the catalog is deliberately small.
This evaluation policy is intentionally separate from the compact display demo,
which selects 10 items from the top 20 candidates.

Do not set $M=K=N$: selecting every candidate removes DPP's subset-selection
effect. Also, do not change $K$ for each NDCG cutoff unless the goal is to
compare separate DPP policies rather than one deployable ranking.

DPP produces a subset rather than an intrinsic ranking. For NDCG, this sample
sorts the selected subset by the original Reranking 1 relevance score. NDCG
measures relevance, so it is reported together with ILD and category coverage
to expose the relevance-diversity tradeoff.
