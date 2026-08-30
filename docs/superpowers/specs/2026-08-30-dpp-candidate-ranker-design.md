# DPP Candidate Ranker Design

## Goal

Extract candidate filtering, relevance scoring, sorting, and recommendation
construction from `app/main.py` into a focused `CandidateRanker`.

## Structure

- `app/ranker.py` defines `CandidateRanker` with
  `rank(model, data, user_id) -> CandidateRanking`.
- `app/schemas/candidate_ranking.py` defines the immutable `CandidateRanking`
  result containing ordered recommendations, scores, and item embeddings.
- `app/schemas/__init__.py` re-exports `CandidateRanking`.
- `app/main.py` constructs the model and data, calls the ranker, applies DPP to
  the returned scores and embeddings, and passes recommendation lists to
  `show()`.

`CandidateRanker` excludes items already observed by the target user, scores
all remaining items in evaluation mode without gradient tracking, sorts them by
descending relevance, and creates matching `Recommendation` values. It raises
`ValueError` when no unseen candidates remain.

## Validation

Focused tests verify seen-item exclusion, descending score order, aligned
recommendations and tensors, and rejection of an empty candidate set. The full
test suite and CLI output verify that DPP reranking behavior remains intact.