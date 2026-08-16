# Matryoshka Representation Learning Design

## Goal

Provide a self-contained PyTorch example of Matryoshka Representation Learning
(MRL) for Japanese text retrieval. The sample trains query-positive passage
pairs with in-batch contrastive learning and demonstrates retrieval quality at
several embedding prefix sizes.

## Scope

The sample is located in `ml_sample/mrl-retrieval/` and
does not require downloading a pretrained model or dataset. It uses a small
character-level tokenizer and a PyTorch Transformer encoder.

The default dataset contains a small set of Japanese query-positive passage
pairs. Users may replace it with JSONL data, where each line has the form:

```json
{"query": "質問", "positive": "関連文書"}
```

## Architecture

- `main.py` defines only `main()`: parses CLI options, prepares data, trains,
  and prints Recall@1 for every configured prefix dimension.
- `model.py` defines only `MatryoshkaEncoder`: a token embedding, positional
  embedding, PyTorch `TransformerEncoder`, masked mean pooling, and L2
  normalization. Its forward method returns a normalized prefix embedding.
- `train.py` defines only `Trainer`: executes batches, computes the loss,
  performs optimizer updates, and reports mean epoch loss.
- `loss.py` defines only `matryoshka_infonce_loss()`: computes a bidirectional,
  in-batch InfoNCE objective independently at every requested prefix dimension
  and returns their mean.
- `preprocess.py` defines only `CharacterTokenizer`: builds a character-level
  vocabulary and transforms text into fixed-length token IDs and attention
  masks.
- `utils.py` contains shared data concerns: built-in examples, JSONL loading,
  `PairDataset`, batch collation, reproducibility helpers, and retrieval
  evaluation.

Except for `utils.py`, every application module owns one public class or
function. Evaluation remains in `utils.py` because it is a data-level helper
over query-passage pairs, rather than a second application workflow.

## Training Behavior

For a batch of $B$ query-passage pairs and prefix dimension $d$, the model
encodes queries $q_i$ and passages $p_i$. It calculates the similarity matrix

$$
S_{ij}^{(d)} = \frac{q_i^{(d)} \cdot p_j^{(d)}}{T}
$$

where $T$ is the temperature. Cross-entropy uses the diagonal as the positive
pair in both query-to-passage and passage-to-query directions. The final loss
averages that bidirectional loss over all selected prefix dimensions, such as
`8,16,32,64`.

The encoder therefore learns representations whose first 8 dimensions are
useful on their own, while progressively longer prefixes retain and improve
retrieval information.

## CLI And Validation

`main.py` supports data path, epoch count, batch size, embedding dimension,
prefix dimensions, random seed, and device options. Defaults run quickly on
CPU using the bundled data.

The program rejects malformed JSONL, missing `query` or `positive` keys, empty
datasets, empty or invalid prefix dimensions, and prefix dimensions larger
than the embedding dimension. Empty text, padding, and unknown characters are
handled by the tokenizer without failing the training loop.

## Tests

Test files mirror the application-file layout exactly:

```text
mrl-retrieval/
  main.py                 tests/test_main.py
  model.py                tests/test_model.py
  train.py                tests/test_train.py
  loss.py                 tests/test_loss.py
  preprocess.py           tests/test_preprocess.py
  utils.py                tests/test_utils.py
```

The tests cover tokenizer behavior, model output shape and normalization,
loss finiteness and backpropagation, trainer parameter updates, data validation,
and a CLI run that prints a metric for each configured prefix dimension.

## Dependencies

Runtime code requires PyTorch. Tests use the Python standard library
`unittest`, so no additional test framework is required.