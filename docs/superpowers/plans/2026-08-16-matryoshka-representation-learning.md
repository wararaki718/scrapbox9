# Matryoshka Representation Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained PyTorch sample that learns Japanese query-passage embeddings with Matryoshka Representation Learning and evaluates prefix-dimension retrieval.

**Architecture:** `CharacterTokenizer` creates fixed-length character tokens. `MatryoshkaEncoder` produces normalized embedding prefixes. `matryoshka_infonce_loss` trains every configured prefix simultaneously, `Trainer` executes optimization, and `main()` drives training and Recall@1 evaluation.

**Tech Stack:** Python 3.11+, PyTorch, standard-library `unittest`.

---

## File Structure

- `ml_sample/matryoshka-representation-learning/main.py`: one `main()` function.
- `ml_sample/matryoshka-representation-learning/model.py`: one `MatryoshkaEncoder` class.
- `ml_sample/matryoshka-representation-learning/train.py`: one `Trainer` class.
- `ml_sample/matryoshka-representation-learning/loss.py`: one `matryoshka_infonce_loss()` function.
- `ml_sample/matryoshka-representation-learning/preprocess.py`: one `CharacterTokenizer` class.
- `ml_sample/matryoshka-representation-learning/utils.py`: data-level helpers.
- `ml_sample/matryoshka-representation-learning/tests/test_*.py`: each app module has a matching test module.

### Task 1: Character Tokenization

**Files:**
- Create: `ml_sample/matryoshka-representation-learning/tests/test_preprocess.py`
- Create: `ml_sample/matryoshka-representation-learning/preprocess.py`

- [ ] **Step 1: Write the failing test**

```python
def test_encode_pads_and_handles_unknown_characters(self) -> None:
    tokenizer = CharacterTokenizer().fit(["猫"])
    token_ids, mask = tokenizer.encode("犬", 4)
    self.assertEqual(token_ids.shape, (4,))
    self.assertEqual(token_ids[0].item(), tokenizer.unknown_id)
    self.assertTrue(torch.equal(mask, torch.tensor([1, 0, 0, 0])))
```

Create the `unittest` module with `torch`, add the sample root to `sys.path`, and import `CharacterTokenizer`.

- [ ] **Step 2: Verify the test fails**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_preprocess -v`

Expected: FAIL because `preprocess` does not exist.

- [ ] **Step 3: Implement the class**

```python
class CharacterTokenizer:
    def __init__(self) -> None:
        self._vocabulary = {"<pad>": 0, "<unk>": 1}

    @property
    def vocabulary_size(self) -> int:
        return len(self._vocabulary)

    @property
    def unknown_id(self) -> int:
        return 1

    def fit(self, texts: list[str]) -> "CharacterTokenizer":
        for text in texts:
            for character in text:
                self._vocabulary.setdefault(character, len(self._vocabulary))
        return self

    def encode(self, text: str, max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        token_ids = [self._vocabulary.get(character, self.unknown_id) for character in text[:max_length]]
        mask = [1] * len(token_ids)
        token_ids.extend([0] * (max_length - len(token_ids)))
        mask.extend([0] * (max_length - len(mask)))
        return torch.tensor(token_ids), torch.tensor(mask)
```

- [ ] **Step 4: Verify the test passes**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_preprocess -v`

Expected: PASS, 1 test.

### Task 2: Data Helpers

**Files:**
- Create: `ml_sample/matryoshka-representation-learning/tests/test_utils.py`
- Create: `ml_sample/matryoshka-representation-learning/utils.py`

- [ ] **Step 1: Write failing tests**

```python
def test_jsonl_requires_query_and_positive(self) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
        file.write('{"query": "質問"}\n')
        file.flush()
        with self.assertRaisesRegex(ValueError, "positive"):
            load_pairs(Path(file.name))

def test_dataset_returns_four_encoded_tensors(self) -> None:
    tokenizer = CharacterTokenizer().fit(["質問", "回答"])
    sample = PairDataset([("質問", "回答")], tokenizer, 4)[0]
    self.assertEqual(set(sample), {"query_ids", "query_mask", "positive_ids", "positive_mask"})

def test_recall_counts_diagonal_match(self) -> None:
    vectors = torch.eye(2)
    self.assertEqual(evaluate_recall_at_one(vectors, vectors), 1.0)
```

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_utils -v`

Expected: FAIL because `utils` does not exist.

- [ ] **Step 3: Implement `utils.py`**

Create `DEFAULT_PAIRS` with four Japanese pairs; `load_pairs(path)` to parse nonblank JSONL lines; `PairDataset`; `create_dataloader`; `set_seed`; and `evaluate_recall_at_one`. Reject empty or malformed datasets and mismatched evaluation shapes with `ValueError`. Dataset items must contain the four tensors named in the test. Recall must use normalized dot products and count an argmax matching each row index.

- [ ] **Step 4: Verify success**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_utils -v`

Expected: PASS, 3 tests.

### Task 3: Encoder

**Files:**
- Create: `ml_sample/matryoshka-representation-learning/tests/test_model.py`
- Create: `ml_sample/matryoshka-representation-learning/model.py`

- [ ] **Step 1: Write the failing test**

```python
def test_encoder_returns_normalized_prefix(self) -> None:
    model = MatryoshkaEncoder(12, 8, 2, 1, 4)
    embedding = model(torch.tensor([[1, 2, 0, 0]]), torch.tensor([[1, 1, 0, 0]]), 4)
    self.assertEqual(embedding.shape, (1, 4))
    self.assertTrue(torch.allclose(embedding.norm(dim=1), torch.ones(1), atol=1e-6))
```

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_model -v`

Expected: FAIL because `model` does not exist.

- [ ] **Step 3: Implement `MatryoshkaEncoder`**

Use token and position `nn.Embedding`, a `batch_first=True` `nn.TransformerEncoder`, masked mean pooling, and a linear projection. `forward(input_ids, attention_mask, dimension)` must validate $1 \le d \le$ embedding width, supply a padding mask to the Transformer, slice the prefix, and L2-normalize it.

- [ ] **Step 4: Verify success**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_model -v`

Expected: PASS, 1 test.

### Task 4: Matryoshka Loss

**Files:**
- Create: `ml_sample/matryoshka-representation-learning/tests/test_loss.py`
- Create: `ml_sample/matryoshka-representation-learning/loss.py`

- [ ] **Step 1: Write the failing test**

```python
def test_loss_backpropagates_at_all_prefixes(self) -> None:
    queries = torch.randn(3, 8, requires_grad=True)
    positives = torch.randn(3, 8, requires_grad=True)
    loss = matryoshka_infonce_loss(queries, positives, [2, 4, 8], 0.1)
    loss.backward()
    self.assertTrue(torch.isfinite(loss))
    self.assertIsNotNone(queries.grad)
```

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_loss -v`

Expected: FAIL because `loss` does not exist.

- [ ] **Step 3: Implement `matryoshka_infonce_loss`**

Validate matching rank-2 tensors, a positive temperature, and valid positive prefix dimensions. For every prefix, normalize query and positive slices; form `logits = queries @ positives.T / temperature`; calculate cross entropy toward diagonal labels for both `logits` and `logits.T`; average directions and dimensions.

- [ ] **Step 4: Verify success**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_loss -v`

Expected: PASS, 1 test.

### Task 5: Training

**Files:**
- Create: `ml_sample/matryoshka-representation-learning/tests/test_train.py`
- Create: `ml_sample/matryoshka-representation-learning/train.py`

- [ ] **Step 1: Write the failing test**

```python
def test_train_epoch_updates_parameters(self) -> None:
    dataset = PairDataset([("東京", "東京の天気"), ("富士山", "富士山の高さ")], tokenizer, 8)
    model = MatryoshkaEncoder(tokenizer.vocabulary_size, 8, 2, 1, 8)
    before = next(model.parameters()).detach().clone()
    Trainer(model, [4, 8], 0.1, 1e-2, torch.device("cpu")).train_epoch(create_dataloader(dataset, 2, False))
    self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
```

Build `tokenizer` from every text in the pairs before the test body.

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_train -v`

Expected: FAIL because `train` does not exist.

- [ ] **Step 3: Implement `Trainer`**

Accept the encoder, dimensions, temperature, learning rate, and device; move the model and build AdamW. `train_epoch` must set training mode, move all four batch tensors, call the model with full embedding width, calculate the MRL loss, perform zero-grad/backward/step, and return mean loss. Reject empty loaders.

- [ ] **Step 4: Verify success**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_train -v`

Expected: PASS, 1 test.

### Task 6: CLI And Documentation

**Files:**
- Create: `ml_sample/matryoshka-representation-learning/tests/test_main.py`
- Create: `ml_sample/matryoshka-representation-learning/main.py`
- Create: `ml_sample/matryoshka-representation-learning/requirements.txt`
- Modify: `ml_sample/matryoshka-representation-learning/README.md`

- [ ] **Step 1: Write the failing CLI test**

```python
def test_main_reports_every_requested_dimension(self) -> None:
    arguments = ["main.py", "--epochs", "1", "--batch-size", "4", "--embedding-dim", "8", "--dimensions", "4,8", "--max-length", "16", "--num-heads", "2", "--num-layers", "1"]
    with patch.object(sys, "argv", arguments), redirect_stdout(self.output):
        main.main()
    self.assertIn("dimension=4 Recall@1=", self.output.getvalue())
    self.assertIn("dimension=8 Recall@1=", self.output.getvalue())
```

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest tests.test_main -v`

Expected: FAIL because `main` does not exist.

- [ ] **Step 3: Implement `main()`**

Parse `--data-path`, `--epochs`, `--batch-size`, `--embedding-dim`, `--dimensions`, `--seed`, `--device`, `--max-length`, `--num-heads`, and `--num-layers`. Parse comma-separated dimensions, use `parser.error` for invalid configurations, train default or JSONL pairs, print `epoch=<n> loss=<value>`, and print `dimension=<d> Recall@1=<score>` after encoding all samples for every requested prefix.

- [ ] **Step 4: Add documentation and requirements**

Put `torch>=2.2,<3` in `requirements.txt`. In the README, document virtual-environment setup, `pip install -r requirements.txt`, default training, JSONL records as `{"query": "...", "positive": "..."}`, available CLI options, and the unittest command.

- [ ] **Step 5: Verify the sample**

Run: `cd ml_sample/matryoshka-representation-learning && python3 -m unittest discover -s tests -v && python3 main.py --epochs 2 --batch-size 4 --embedding-dim 16 --dimensions 4,8,16 --max-length 32 --num-heads 2 --num-layers 1`

Expected: all tests pass; two loss lines and Recall@1 for dimensions 4, 8, and 16 are printed.

- [ ] **Step 6: Inspect the diff**

Run: `cd /Users/wararaki/workspace/scrapbox9 && git diff --check && git status --short`

Expected: only Matryoshka sample and its design/plan files are changed.