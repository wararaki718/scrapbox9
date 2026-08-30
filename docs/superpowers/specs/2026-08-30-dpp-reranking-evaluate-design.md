# DPP Reranking Evaluate Design

## Goal

Add an offline evaluation example that compares a relevance reranker with DPP using deterministic dummy judgments and documents how DPP candidate and selection sizes relate to evaluation cutoffs.

## Package Structure

```text
app/evaluate/
  __init__.py
  evaluator.py
  metrics/
    __init__.py
    ndcg.py
    intra_list_diversity.py
    category_coverage.py

tests/evaluate/
  __init__.py
  test_evaluator.py
  metrics/
    __init__.py
    test_ndcg.py
    test_intra_list_diversity.py
    test_category_coverage.py
```

Each metric module owns one public function and its validation. `Evaluator` composes the metrics and prepares comparable rankings; it does not own metric formulas.

## Metrics

`ndcg_at_k(item_ids, relevance_by_item_id, k)` computes graded NDCG with gain $2^{rel}-1$ and logarithmic discount. Missing item judgments are treated as zero relevance. It rejects non-positive cutoffs and returns zero when the ideal DCG is zero.

`intra_list_diversity_at_k(item_ids, embedding_by_item_id, k)` computes the mean pairwise cosine distance among the first $k$ items. It validates that all selected items have finite, same-dimensional embeddings and returns zero for fewer than two items.

`category_coverage_at_k(item_ids, category_by_item_id, k)` computes the number of distinct categories represented among the first $k$ items. It rejects missing category metadata and non-positive cutoffs.

## Evaluation Flow

The existing model and `CandidateRanker` produce the Reranking 1 relevance order over all unseen candidates. Deterministic dummy graded judgments are generated from item metadata and IDs so the example requires no external dataset.

Evaluation uses:

- DPP candidate size $M$: all 27 unseen candidates in this sample.
- DPP selection size $K$: 20.
- NDCG cutoffs $N$: 5, 10, and 20.

DPP selects 20 items from the 27-item pool. Those selected items are then sorted by the original Reranking 1 relevance score before NDCG and diversity metrics are computed. This assigns DPP responsibility for subset selection and Reranking 1 responsibility for ordering within the subset.

The evaluator returns rows for `Reranking 1` and `DPP`, each containing NDCG@5, NDCG@10, NDCG@20, ILD@20, and category coverage@20. `main()` prints this evaluation table after the existing recommendation tables.

## DPP Range Guidance

Use distinct symbols for three independent sizes:

- $M$: number of candidates passed to DPP.
- $K$: number of items selected by DPP.
- $N$: evaluation cutoff in NDCG@$N$.

For one deployable ranking evaluated at multiple cutoffs, keep DPP configuration fixed and use $M > K \ge \max(N)$. For NDCG@5, @10, and @20, one suitable configuration is $M=100$ and $K=20$; this sample scales it to $M=27$ and $K=20$ because only 27 unseen dummy candidates exist.

Do not set $M=K=N$: selecting every DPP candidate removes the subset-selection effect. Do not rerun DPP with a different $K$ for each NDCG cutoff unless the experiment explicitly compares separate DPP policies. NDCG should be paired with a diversity metric such as ILD or category coverage because DPP may trade some relevance for reduced redundancy.

## Validation

Unit tests cover each metric formula and invalid input. Evaluator tests verify fixed cutoffs, a 20-item DPP ranking, relevance reordering after DPP selection, and both result rows. The main integration test verifies that the evaluation table is printed without changing the existing 20-item relevance and 10-item display DPP sections.
