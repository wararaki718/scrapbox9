# MRL Image Dataset Sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible, independently configurable CIFAR-10 train and test sampling for fast local validation.

**Architecture:** `parse_args()` supplies positive `train_samples` and `test_samples` values with validation-oriented defaults. `create_dataloaders()` constructs CIFAR-10 as before, then creates deterministic `Subset` instances using a locally seeded `torch.Generator`. `main()` passes the CLI values and seed to the loader factory; the README exposes defaults and the full-dataset override.

**Tech Stack:** Python 3.11+, PyTorch, torchvision, standard-library `unittest`.

---

## File Structure

- Modify: `ml_sample/mrl-image/app/args.py`: add and validate sample-count options.
- Modify: `ml_sample/mrl-image/app/dataset.py`: deterministically select sampled train and test subsets.
- Modify: `ml_sample/mrl-image/app/main.py`: pass CLI sample configuration to loader creation.
- Modify: `ml_sample/mrl-image/tests/test_args.py`: cover sample-count defaults and rejection.
- Modify: `ml_sample/mrl-image/tests/test_dataset.py`: cover sampled size, reproducibility, and oversized requests with mocked CIFAR-10.
- Modify: `ml_sample/mrl-image/tests/test_main.py`: verify CLI values are forwarded.
- Modify: `ml_sample/mrl-image/README.md`: document default sampling and full-dataset execution.

### Task 1: Add Sample-Count CLI Options

**Files:**
- Modify: `ml_sample/mrl-image/app/args.py`
- Modify: `ml_sample/mrl-image/tests/test_args.py`

- [ ] **Step 1: Write failing argument tests**

```python
def test_parse_args_defaults_to_validation_sample_sizes(self) -> None:
    args = parse_args([])
    self.assertEqual(args.train_samples, 1000)
    self.assertEqual(args.test_samples, 200)

def test_parse_args_rejects_non_positive_sample_sizes(self) -> None:
    for option in ("--train-samples", "--test-samples"):
        with self.subTest(option=option), self.assertRaises(SystemExit):
            parse_args([option, "0"])
```

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_args -v`

Expected: FAIL because `Namespace` has no `train_samples` or `test_samples` attributes.

- [ ] **Step 3: Add parser options**

```python
parser.add_argument("--train-samples", type=_positive_int, default=1000)
parser.add_argument("--test-samples", type=_positive_int, default=200)
```

Place these beside `--batch-size`; reuse the existing `_positive_int` validator.

- [ ] **Step 4: Verify success**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_args -v`

Expected: PASS with the new default and invalid-value tests.

### Task 2: Sample CIFAR-10 Splits Deterministically

**Files:**
- Modify: `ml_sample/mrl-image/app/dataset.py`
- Modify: `ml_sample/mrl-image/tests/test_dataset.py`

- [ ] **Step 1: Write failing data-loader tests**

Use mocked CIFAR-10 datasets containing ten indexed tensor images in each split. Add a test that calls:

```python
first_train, first_test = create_dataloaders(Path("data"), 2, 4, 3, seed=42)
second_train, second_test = create_dataloaders(Path("data"), 2, 4, 3, seed=42)
self.assertEqual(len(first_train.dataset), 4)
self.assertEqual(len(first_test.dataset), 3)
self.assertEqual(first_train.dataset.indices, second_train.dataset.indices)
self.assertEqual(first_test.dataset.indices, second_test.dataset.indices)
```

Add a separate test that requests 11 samples from a mocked ten-item split and asserts `ValueError` includes `sample_count`.

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_dataset -v`

Expected: FAIL because `create_dataloaders()` does not accept sample-count and seed arguments.

- [ ] **Step 3: Implement deterministic subsets**

```python
def _sample_dataset(dataset: Dataset, sample_count: int, seed: int) -> Subset:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if sample_count > len(dataset):
        raise ValueError("sample_count must not exceed dataset size")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_count].tolist()
    return Subset(dataset, indices)

def create_dataloaders(
    data_dir: Path,
    batch_size: int,
    train_samples: int,
    test_samples: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
```

Preserve the existing validation, CIFAR-10 transforms, download behavior, and train/test `shuffle` flags. Create the test subset using `seed + 1` so the two split selections are deterministic but independently generated. Pass each sampled subset to its existing DataLoader.

- [ ] **Step 4: Verify success**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_dataset -v`

Expected: PASS; no network requests occur because CIFAR-10 remains mocked.

### Task 3: Wire Sampling Through the CLI and Document It

**Files:**
- Modify: `ml_sample/mrl-image/app/main.py`
- Modify: `ml_sample/mrl-image/tests/test_main.py`
- Modify: `ml_sample/mrl-image/README.md`

- [ ] **Step 1: Update the failing integration test**

Update the patched `create_dataloaders` assertion to require all forwarded values:

```python
create_dataloaders.assert_called_once_with(
    Path("data"),
    batch_size=64,
    train_samples=1000,
    test_samples=200,
    seed=42,
)
```

- [ ] **Step 2: Verify failure**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_main -v`

Expected: FAIL because `main()` still calls the loader factory with only data directory and batch size.

- [ ] **Step 3: Forward sample configuration and update documentation**

```python
train_loader, test_loader = create_dataloaders(
    args.data_dir,
    batch_size=args.batch_size,
    train_samples=args.train_samples,
    test_samples=args.test_samples,
    seed=args.seed,
)
```

Add `--train-samples` and `--test-samples` to the README option list. State that defaults are 1,000 training and 200 test images, selected reproducibly from the given seed. Include this full-dataset command:

```bash
make run ARGS="--train-samples 50000 --test-samples 10000"
```

- [ ] **Step 4: Verify focused and complete behavior**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_main -v && python -m unittest discover -s tests -v`

Expected: all tests pass without downloading CIFAR-10.

- [ ] **Step 5: Validate final change set**

Run: `cd /Users/wararaki/workspace/scrapbox9 && git diff --check && git status --short`

Expected: no whitespace errors; only sampling implementation, its tests, README, and the sampling design/plan documentation are changed.