# DPP Reranking Design

## Goal

Provide a self-contained PyTorch example that trains a simple two-tower model
on sample implicit-feedback data, generates relevance-ranked candidates, and
reranks them with a Determinantal Point Process (DPP) to balance relevance and
diversity.

The example must make the effect of reranking visible by printing the ordinary
relevance top-k list beside the DPP top-k list, including each item's category
and relevance score.

## Scope

The sample is located in `ml_sample/dpp-reranking/`. It uses deterministic,
in-memory sample data and depends only on PyTorch. It includes model training,
candidate generation, Fast Greedy MAP DPP reranking, a command-line demo, and
unit tests.

The scope excludes production datasets, approximate nearest-neighbor search,
model persistence, distributed training, stochastic k-DPP sampling, and exact
MAP enumeration. Item categories are used to construct understandable sample
preferences and to explain the output; DPP similarity itself comes from the
item tower's learned embeddings.

## Architecture

- `app/config.py` defines the frozen `TrainConfig` dataclass containing the
  embedding dimension, epoch count, learning rate, and random seed.
- `app/data.py` defines item, interaction, and `Recommendation` data structures
  and constructs a deterministic sample dataset with several item categories
  and user preferences. A `Recommendation` contains an item ID, category, and
  relevance score.
- `app/user_tower.py` defines `UserTower`, which maps user IDs to normalized
  embeddings.
- `app/item_tower.py` defines `ItemTower`, which maps item IDs to normalized
  embeddings.
- `app/model.py` defines `TwoTowerModel`, which composes both towers and scores
  user-item pairs with embedding dot products.
- `app/train.py` defines `train()`, which samples negative items and optimizes
  the model with Bayesian Personalized Ranking (BPR) loss. It returns the mean
  loss for each epoch.
- `app/reranker.py` defines `KernelBuilder`, `GreedyMapSelector`, and
  `DPPReranker`. Algorithmic steps are represented by these focused classes
  instead of being hidden in private methods.
- `app/utils.py` defines `show()`, which receives ordinary and reranked result
  lists and prints their item IDs, categories, and relevance scores.
- `app/main.py` composes data creation, training, candidate scoring, reranking,
  and the call to `show()`. It does not implement output formatting or DPP
  matrix operations.

Tests mirror these responsibilities under `tests/`. `Makefile` provides
`make run` and `make test`, while `requirements.txt` lists PyTorch.

## Sample Data And Training

The in-memory dataset contains multiple users and items spread across several
categories. Each user has positive interactions concentrated in preferred
categories, with enough overlap for the item tower to learn useful similarity.
Known positive items are excluded when generating recommendations for the demo
user.

For each positive interaction `(user_id, positive_item_id)`, training samples
one item with which that user has not interacted. Given positive and negative
scores $s(u, i^+)$ and $s(u, i^-)$, the loss is

$$
-\log \sigma\left(s(u, i^+) - s(u, i^-)\right).
$$

Training uses Adam and a locally seeded random-number generator. The loop is
kept intentionally small and direct instead of introducing a `DataLoader`.
The same configuration therefore produces reproducible training and output.

## DPP Reranking

`DPPReranker.rerank(scores, item_embeddings, top_k)` is the public reranking
entry point and returns indices into the input candidate list. It delegates to
two independently testable collaborators:

1. `KernelBuilder.build(scores, item_embeddings)` converts relevance scores
  into positive quality weights $q_i = \exp(\alpha r_i)$, computes cosine
  similarity from normalized item embeddings $\phi_i$, and builds

   $$
   L_{ij} = q_i\left(\phi_i^\top\phi_j\right)q_j.
   $$

  This is a quality-weighted Gram matrix and is positive semidefinite. The
  quality scale $\alpha$ is supplied to `KernelBuilder` as `quality_scale` and
  defaults to `1.0`; it must be finite and non-negative. A small diagonal
  epsilon is added for numerical stability.
2. `GreedyMapSelector.select(kernel, top_k)` performs Fast Greedy MAP selection.
   It incrementally maintains the Cholesky factors and diagonal residuals used
   to calculate each candidate's marginal gain. It does not recompute a matrix
   determinant for every candidate and iteration. Its complexity is
   approximately $O(Nk^2)$ for $N$ candidates and $k$ selected items.

Ties are resolved by original candidate order. Selection stops early if every
remaining diagonal residual is at or below the numerical tolerance. With the
stability epsilon and valid input, the normal demo returns exactly `top_k`
indices.

`KernelBuilder`, `GreedyMapSelector`, and `DPPReranker` live together in
`app/reranker.py` because they form one small DPP component. Construction
allows a builder and selector to be supplied to `DPPReranker`, keeping the
orchestration independent from either numerical implementation.

## Validation And Error Handling

The towers reject invalid vocabulary sizes and embedding dimensions. Training
rejects empty interactions, users without a valid negative item, and invalid
configuration values.

The reranking component rejects mismatched candidate counts, tensors with
invalid ranks, non-finite values, non-square kernels, non-positive `top_k`, and
`top_k` values greater than the candidate count. These conditions raise
`ValueError` with messages that identify the invalid input. Numerical
intermediates use PyTorch tensors on the input embedding device and preserve a
floating-point dtype suitable for the matrix calculations.

## Output

The CLI trains one model, scores unseen items for a selected sample user, and
forms a candidate pool in descending relevance order. It compares the ordinary
top-k list with the DPP-selected top-k list.

`show()` owns all presentation. Its ordinary and reranked arguments are
sequences of `Recommendation` values, and it also accepts an output stream.
This makes formatting testable without patching global standard output. Each
displayed row contains rank, item ID, category, and relevance score. The DPP
section preserves DPP selection order.

## Tests

Tests use Python's `unittest` and fixed random seeds.

- `UserTower` and `ItemTower` return embeddings with the configured shape and
  unit norm.
- `TwoTowerModel` returns the expected score shape and propagates gradients to
  both towers.
- BPR training lowers loss and makes known positive items score above sampled
  negative items on the sample dataset.
- `KernelBuilder` returns a finite, symmetric, positive-semidefinite matrix and
  gives higher-quality candidates larger diagonal values.
- `GreedyMapSelector` returns unique indices, respects `top_k`, resolves ties
  deterministically, and selects complementary candidates in a controlled
  kernel example.
- `DPPReranker` delegates kernel construction and selection and rejects invalid
  shapes, values, and selection counts.
- `show()` formats both recommendation lists through the supplied stream.
- A lightweight `main()` test runs the complete training, scoring, reranking,
  and display workflow.

Implementation follows test-driven development: each production behavior is
introduced by a focused failing test, followed by the minimum implementation
needed to pass it.