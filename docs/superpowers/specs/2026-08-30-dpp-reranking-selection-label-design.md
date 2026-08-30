# DPP Reranking Selection Label Design

## Goal

Compare a 20-item relevance ranking with a 10-item DPP selection and clearly label which relevance-ranked items were selected by DPP.

## Selection Flow

`CandidateRanker` continues to score and sort all unseen candidates. `main()` takes the first 20 recommendations, scores, and item embeddings as the comparison pool. `DPPReranker` receives only those 20 score and embedding rows and selects 10 indices.

The selected indices therefore always refer to the displayed 20-item relevance ranking. The DPP output contains exactly 10 recommendations and is guaranteed to be a subset of the relevance output.

## Display

`show()` derives the selected item IDs from its `reranked` argument and adds a `label` column to both tables.

- A relevance row selected by DPP displays `reranked`.
- A relevance row not selected by DPP displays `-`.
- Every DPP row displays `reranked`.

This keeps selection metadata out of the immutable `Recommendation` value object because the label belongs to a particular reranking run rather than to the item recommendation itself.

## Validation

The main integration test will verify 20 relevance rows, 10 DPP rows, and that all DPP item IDs occur in the relevance table. The output formatting test will verify selected and unselected labels. Existing ranking and reranking algorithms remain unchanged.
