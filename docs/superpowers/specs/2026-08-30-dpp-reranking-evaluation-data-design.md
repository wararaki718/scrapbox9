# DPP Reranking Evaluation Data Design

## Goal

Use separate interaction data for model training and reranking evaluation, and display approximately twenty recommendations for both relevance and DPP rankings.

## Data Design

The training and reranking datasets share one catalog of 30 items and the same three-user ID space. This is required because the current two-tower model learns ID-based embeddings and cannot score item or user IDs absent from its embedding tables.

`app.data` will expose:

- `create_training_data()`: 30 catalog items plus interactions used only by BPR training.
- `create_reranking_data()`: the same 30 catalog items plus a distinct set of interactions representing items already known at reranking time.

Both functions return `SampleData`, preserving the existing interfaces used by `train` and `CandidateRanker`. A private catalog factory may be shared to guarantee matching item IDs and metadata without duplicating construction logic.

For user 0, the reranking data contains three known items. Candidate ranking therefore has 27 unseen items, enough to select and print 20.

## Application Flow

`main()` will:

1. Create training and reranking data independently.
2. Initialize the model using the training data dimensions.
3. Train only with training interactions.
4. Build candidates only from reranking data for user 0.
5. Select `top_k = 20` through `DPPReranker`.
6. Print 20 relevance-ranked and 20 DPP-ranked recommendations.

## Validation

Data tests will verify that both datasets use a valid, matching 30-item catalog and have distinct interactions. The main integration test will verify that each output section contains exactly 20 recommendation rows. Existing model, training, candidate ranking, and DPP behavior must remain unchanged.
