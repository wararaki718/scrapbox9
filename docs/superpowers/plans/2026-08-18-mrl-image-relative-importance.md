# MRL Image Relative Importance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users assign a relative loss importance to each MRL prefix dimension, defaulting to the Appendix A standard of one for every prefix.

**Architecture:** `parse_args()` parses optional comma-separated weights after dimensions are known. `Trainer` stores the validated weight tuple and passes it to `matryoshka_cross_entropy()`, which computes the paper's weighted sum rather than an unweighted mean. `main()` forwards the parsed configuration, and the README documents the option.

**Tech Stack:** Python 3.11+, PyTorch, torchvision, standard-library `unittest`.

---

## File Structure

- Modify: `ml_sample/mrl-image/app/args.py`: parse and validate `--loss-weights`.
- Modify: `ml_sample/mrl-image/app/loss.py`: validate weights and calculate weighted cross-entropy sum.
- Modify: `ml_sample/mrl-image/app/train.py`: retain and forward relative importance.
- Modify: `ml_sample/mrl-image/app/main.py`: pass parsed weights to `Trainer`.
- Modify: `ml_sample/mrl-image/tests/test_args.py`: cover defaults and invalid weight configuration.
- Modify: `ml_sample/mrl-image/tests/test_loss.py`: cover weighted numerical result and invalid weights.
- Modify: `ml_sample/mrl-image/tests/test_train.py`: prove Trainer forwards weights to the loss function.
- Modify: `ml_sample/mrl-image/tests/test_main.py`: prove CLI defaults are supplied to `Trainer`.
- Modify: `ml_sample/mrl-image/README.md`: document syntax and low-dimensional emphasis example.

### Task 1: Parse Relative Importance

**Files:**
- Modify: `ml_sample/mrl-image/app/args.py`
- Modify: `ml_sample/mrl-image/tests/test_args.py`

- [ ] **Step 1: Write failing parser tests**

```python
def test_parse_args_defaults_loss_weights_to_one_per_dimension(self) -> None:
    args = parse_args(["--dimensions", "4,8"])
    self.assertEqual(args.loss_weights, [1.0, 1.0])

def test_parse_args_accepts_loss_weights(self) -> None:
    args = parse_args(["--dimensions", "4,8", "--loss-weights", "2,0.5"])
    self.assertEqual(args.loss_weights, [2.0, 0.5])

def test_parse_args_rejects_invalid_loss_weights(self) -> None:
    for value in ("1", "1,2,3", "1,-1", "1,nan", "1,inf", ""):
        with self.subTest(value=value), self.assertRaises(SystemExit):
            parse_args(["--dimensions", "4,8", "--loss-weights", value])
```

- [ ] **Step 2: Verify the tests fail**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_args -v`

Expected: FAIL because `loss_weights` is not present on `argparse.Namespace`.

- [ ] **Step 3: Add parsing and validation**

Add `parser.add_argument("--loss-weights")`. After dimensions are parsed, set default weights with:

```python
if args.loss_weights is None:
    args.loss_weights = [1.0] * len(args.dimensions)
else:
    try:
        args.loss_weights = [float(component) for component in args.loss_weights.split(",")]
    except ValueError:
        parser.error("loss-weights must be comma-separated numbers")
    if len(args.loss_weights) != len(args.dimensions):
        parser.error("loss-weights must have one value per dimension")
    if any(not math.isfinite(weight) or weight < 0 for weight in args.loss_weights):
        parser.error("loss-weights must be finite and non-negative")
```

- [ ] **Step 4: Verify the tests pass**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_args -v`

Expected: PASS with default, valid, and invalid loss-weight coverage.

### Task 2: Calculate the Weighted MRL Objective

**Files:**
- Modify: `ml_sample/mrl-image/app/loss.py`
- Modify: `ml_sample/mrl-image/tests/test_loss.py`

- [ ] **Step 1: Write failing weighted-loss tests**

Replace the mean expectation with the weighted sum and add invalid-weight cases:

```python
weights = [2.0, 0.5]
expected = sum(
    weight * F.cross_entropy(logits, labels)
    for weight, logits in zip(weights, logits_by_dimension.values(), strict=True)
)
self.assertTrue(torch.allclose(matryoshka_cross_entropy(logits_by_dimension, labels, weights), expected))

for weights in ([], [1.0], [1.0, 2.0, 3.0], [1.0, -1.0], [1.0, float("nan")]):
    with self.subTest(weights=weights), self.assertRaises(ValueError):
        matryoshka_cross_entropy(logits_by_dimension, labels, weights)
```

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_loss -v`

Expected: FAIL because the loss function does not accept weights.

- [ ] **Step 3: Implement the weighted sum**

Change the signature to:

```python
def matryoshka_cross_entropy(
    logits_by_dimension: Mapping[int, Tensor], labels: Tensor, relative_importance: Sequence[float]
) -> Tensor:
```

Validate that `relative_importance` contains exactly one finite non-negative value per logit tensor. Iterate with `zip(relative_importance, logits_by_dimension.values(), strict=True)`, validate each tensor as currently implemented, append `weight * functional.cross_entropy(logits, labels)`, and return `torch.stack(losses).sum()`.

- [ ] **Step 4: Verify success**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_loss -v`

Expected: PASS; the numerical test demonstrates a weighted sum rather than a mean.

### Task 3: Wire the Trainer, CLI, and Documentation

**Files:**
- Modify: `ml_sample/mrl-image/app/train.py`
- Modify: `ml_sample/mrl-image/app/main.py`
- Modify: `ml_sample/mrl-image/tests/test_train.py`
- Modify: `ml_sample/mrl-image/tests/test_main.py`
- Modify: `ml_sample/mrl-image/README.md`

- [ ] **Step 1: Write failing forwarding tests**

Add a `unittest.mock.patch("app.train.matryoshka_cross_entropy", wraps=...)` test that creates `Trainer(model, cpu, 0.01, [2.0, 1.0])`, runs one batch, and asserts the mock's third positional argument is `(2.0, 1.0)`.

In `test_main.py`, patch `app.main.Trainer`, call `main()` using existing mocked loaders, and assert its constructor received `loss_weights=[1.0, 1.0]` after the model and device arguments.

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_train tests.test_main -v`

Expected: FAIL because `Trainer` does not accept relative importance and `main()` does not pass it.

- [ ] **Step 3: Pass weights end-to-end**

Extend the trainer constructor:

```python
def __init__(
    self, model: MatryoshkaImageClassifier, device: torch.device,
    learning_rate: float, relative_importance: Sequence[float]
) -> None:
    if len(relative_importance) != len(model.dimensions):
        raise ValueError("relative_importance must have one value per dimension")
    if any(not math.isfinite(weight) or weight < 0 for weight in relative_importance):
        raise ValueError("relative_importance must be finite and non-negative")
    self.relative_importance = tuple(relative_importance)
```

Pass `self.relative_importance` as the third argument to `matryoshka_cross_entropy()` in `train_epoch()`. In `main()`, construct the trainer with `args.loss_weights` as the fourth argument.

Update the README option list and add:

```sh
make run ARGS="--dimensions 8,16,32,64 --loss-weights 2,1,1,1"
```

State that omitted weights default to `1.0` for each configured dimension.

- [ ] **Step 4: Verify complete behavior**

Run: `cd ml_sample/mrl-image && python -m unittest discover -s tests -v`

Expected: all tests pass without downloading CIFAR-10.

- [ ] **Step 5: Validate the change set**

Run: `cd /Users/wararaki/workspace/scrapbox9 && git diff --check && git status --short`

Expected: no whitespace errors; only weighted-loss implementation, tests, README, and the relevant specification/plan files are changed.