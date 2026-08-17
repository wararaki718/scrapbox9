# Matryoshka Image Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CIFAR-10 PyTorch sample that trains one CNN with Matryoshka prefix classifiers and reports accuracy at each configured embedding dimension.

**Architecture:** `create_dataloaders()` owns CIFAR-10 download and transformations. `MatryoshkaImageClassifier` produces a maximum-width embedding and maps each requested prefix through a dedicated classifier. `matryoshka_cross_entropy()` averages the prefix losses, while `Trainer` owns training and per-prefix evaluation. `main()` composes these pieces through the command line.

**Tech Stack:** Python 3.11+, PyTorch, torchvision, standard-library `unittest`.

---

## File Structure

- `ml_sample/mrl-image/app/args.py`: CLI parsing and validation.
- `ml_sample/mrl-image/app/dataset.py`: CIFAR-10 transformations and data loaders.
- `ml_sample/mrl-image/app/model.py`: lightweight CNN and prefix classifiers.
- `ml_sample/mrl-image/app/loss.py`: mean cross-entropy across prefixes.
- `ml_sample/mrl-image/app/train.py`: one-epoch optimization and evaluation.
- `ml_sample/mrl-image/app/main.py`: application composition and metric output.
- `ml_sample/mrl-image/tests/test_*.py`: unit and lightweight integration tests.
- `ml_sample/mrl-image/Makefile`, `requirements.txt`, `README.md`: execution interface and documentation.

### Task 1: Create CLI Parsing With Validation

**Files:**
- Create: `ml_sample/mrl-image/app/__init__.py`
- Create: `ml_sample/mrl-image/app/args.py`
- Create: `ml_sample/mrl-image/tests/__init__.py`
- Create: `ml_sample/mrl-image/tests/test_args.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from app.args import parse_args


class ArgsTests(unittest.TestCase):
    def test_parses_dimensions(self) -> None:
        args = parse_args(["--embedding-dim", "16", "--dimensions", "4,8,16"])
        self.assertEqual(args.dimensions, [4, 8, 16])

    def test_rejects_duplicate_prefix_dimensions(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--dimensions", "8,8"])
```

- [ ] **Step 2: Verify the tests fail**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_args -v`

Expected: FAIL because the `app` package does not exist.

- [ ] **Step 3: Implement `parse_args()`**

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dimensions", default="8,16,32,64")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    try:
        args.dimensions = [int(value) for value in args.dimensions.split(",")]
    except ValueError:
        parser.error("dimensions must be comma-separated integers")
    if (args.epochs <= 0 or args.batch_size <= 0 or args.embedding_dim <= 0
            or args.learning_rate <= 0 or not args.dimensions
            or min(args.dimensions) <= 0 or max(args.dimensions) > args.embedding_dim
            or len(set(args.dimensions)) != len(args.dimensions)):
        parser.error("invalid training or prefix dimension configuration")
    return args
```

Import `argparse`, `Sequence`, `Path`, and `torch` at module scope.

- [ ] **Step 4: Verify the tests pass**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_args -v`

Expected: PASS, 2 tests.

### Task 2: Add CIFAR-10 Data Loaders

**Files:**
- Create: `ml_sample/mrl-image/app/dataset.py`
- Create: `ml_sample/mrl-image/tests/test_dataset.py`

- [ ] **Step 1: Write the failing loader test**

```python
@patch("app.dataset.datasets.CIFAR10")
def test_creates_normalized_image_loaders(self, cifar10: Mock) -> None:
    cifar10.side_effect = [TensorDataset(torch.zeros(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])),
                            TensorDataset(torch.zeros(2, 3, 32, 32), torch.tensor([4, 5]))]
    train_loader, test_loader = create_dataloaders(Path("data"), batch_size=2)
    images, labels = next(iter(train_loader))
    self.assertEqual(images.shape, (2, 3, 32, 32))
    self.assertEqual(labels.shape, (2,))
    self.assertEqual(len(test_loader.dataset), 2)
```

Import `Path`, `Mock`, `patch`, `TensorDataset`, and `create_dataloaders` in the test module.

- [ ] **Step 2: Verify the test fails**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_dataset -v`

Expected: FAIL because `app.dataset` does not exist.

- [ ] **Step 3: Implement the loader factory**

```python
def create_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )
```

Use `torch.utils.data.DataLoader` and `torchvision.datasets, transforms` imports. Reject a non-positive `batch_size` with `ValueError` before constructing the datasets.

- [ ] **Step 4: Verify the test passes**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_dataset -v`

Expected: PASS, 1 test, with no network request.

### Task 3: Implement the MRL Classifier and Loss

**Files:**
- Create: `ml_sample/mrl-image/app/model.py`
- Create: `ml_sample/mrl-image/app/loss.py`
- Create: `ml_sample/mrl-image/tests/test_model.py`
- Create: `ml_sample/mrl-image/tests/test_loss.py`

- [ ] **Step 1: Write failing model and loss tests**

```python
def test_model_returns_logits_for_each_prefix(self) -> None:
    model = MatryoshkaImageClassifier(embedding_dim=16, dimensions=[4, 8, 16])
    logits = model(torch.randn(3, 3, 32, 32))
    self.assertEqual(set(logits), {4, 8, 16})
    self.assertEqual(logits[4].shape, (3, 10))

def test_loss_backpropagates_through_all_prefixes(self) -> None:
    model = MatryoshkaImageClassifier(embedding_dim=16, dimensions=[4, 8, 16])
    loss = matryoshka_cross_entropy(model(torch.randn(3, 3, 32, 32)), torch.tensor([0, 1, 2]))
    loss.backward()
    self.assertTrue(torch.isfinite(loss))
    self.assertTrue(all(head.weight.grad is not None for head in model.classifiers.values()))
```

- [ ] **Step 2: Verify the tests fail**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_model tests.test_loss -v`

Expected: FAIL because the model and loss modules do not exist.

- [ ] **Step 3: Implement the model and loss**

```python
class MatryoshkaImageClassifier(nn.Module):
    def __init__(self, embedding_dim: int, dimensions: Sequence[int]) -> None:
        super().__init__()
        if not dimensions or min(dimensions) <= 0 or max(dimensions) > embedding_dim:
            raise ValueError("dimensions must fit within embedding_dim")
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Linear(64, embedding_dim)
        self.classifiers = nn.ModuleDict({str(dimension): nn.Linear(dimension, 10) for dimension in dimensions})

    def forward(self, images: torch.Tensor) -> dict[int, torch.Tensor]:
        embedding = self.projection(self.features(images).flatten(1))
        return {int(dimension): classifier(embedding[:, :int(dimension)])
                for dimension, classifier in self.classifiers.items()}

def matryoshka_cross_entropy(logits_by_dimension: Mapping[int, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    if not logits_by_dimension:
        raise ValueError("at least one prefix logit tensor is required")
    return torch.stack([functional.cross_entropy(logits, labels) for logits in logits_by_dimension.values()]).mean()
```

Validate that every logit tensor is rank 2, has ten columns, and has the same batch length as `labels`; raise `ValueError` otherwise.

- [ ] **Step 4: Verify the tests pass**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_model tests.test_loss -v`

Expected: PASS, 2 tests.

### Task 4: Add Training and Prefix Evaluation

**Files:**
- Create: `ml_sample/mrl-image/app/train.py`
- Create: `ml_sample/mrl-image/tests/test_train.py`

- [ ] **Step 1: Write failing tests**

```python
def test_train_epoch_updates_parameters(self) -> None:
    model = MatryoshkaImageClassifier(16, [4, 8, 16])
    before = model.projection.weight.detach().clone()
    loader = DataLoader(TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3])), batch_size=2)
    Trainer(model, torch.device("cpu"), 1e-3).train_epoch(loader)
    self.assertFalse(torch.equal(before, model.projection.weight.detach()))

def test_evaluate_returns_accuracy_for_every_prefix(self) -> None:
    loader = DataLoader(TensorDataset(torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])), batch_size=3)
    results = Trainer(MatryoshkaImageClassifier(8, [4, 8]), torch.device("cpu"), 1e-3).evaluate(loader)
    self.assertEqual(set(results), {4, 8})
    self.assertTrue(all(0.0 <= value <= 1.0 for value in results.values()))
```

- [ ] **Step 2: Verify the tests fail**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_train -v`

Expected: FAIL because `Trainer` does not exist.

- [ ] **Step 3: Implement `Trainer`**

```python
class Trainer:
    def __init__(self, model: MatryoshkaImageClassifier, device: torch.device, learning_rate: float) -> None:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        losses = []
        for images, labels in loader:
            logits = self.model(images.to(self.device))
            loss = matryoshka_cross_entropy(logits, labels.to(self.device))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        if not losses:
            raise ValueError("loader must not be empty")
        return sum(losses) / len(losses)
```

Implement `evaluate()` with `model.eval()` and `torch.no_grad()`. Track correct and total labels separately for each key returned by the model, then return `{dimension: correct / total}`. Reject an empty loader with `ValueError`.

- [ ] **Step 4: Verify the tests pass**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_train -v`

Expected: PASS, 2 tests.

### Task 5: Compose the CLI and Document Usage

**Files:**
- Create: `ml_sample/mrl-image/app/main.py`
- Create: `ml_sample/mrl-image/tests/test_main.py`
- Create: `ml_sample/mrl-image/Makefile`
- Create: `ml_sample/mrl-image/requirements.txt`
- Create: `ml_sample/mrl-image/README.md`

- [ ] **Step 1: Write the failing integration test**

```python
@patch("app.main.create_dataloaders")
def test_main_reports_every_prefix_dimension(self, create_dataloaders: Mock) -> None:
    dataset = TensorDataset(torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    create_dataloaders.return_value = (DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=2))
    with patch.object(sys, "argv", ["main.py", "--epochs", "1", "--embedding-dim", "8", "--dimensions", "4,8"]), redirect_stdout(io.StringIO()) as output:
        main()
    self.assertIn("dimension=4 accuracy=", output.getvalue())
    self.assertIn("dimension=8 accuracy=", output.getvalue())
```

- [ ] **Step 2: Verify the test fails**

Run: `cd ml_sample/mrl-image && python -m unittest tests.test_main -v`

Expected: FAIL because `app.main` does not exist.

- [ ] **Step 3: Implement the entry point**

```python
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    train_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size)
    trainer = Trainer(MatryoshkaImageClassifier(args.embedding_dim, args.dimensions), torch.device(args.device), args.learning_rate)
    for epoch in range(1, args.epochs + 1):
        print(f"epoch={epoch} loss={trainer.train_epoch(train_loader):.4f}")
    for dimension, accuracy in trainer.evaluate(test_loader).items():
        print(f"dimension={dimension} accuracy={accuracy:.4f}")
```

Add the standard `if __name__ == "__main__": main()` guard.

- [ ] **Step 4: Add the execution interface**

Create the Makefile:

```make
PYTHON ?= python
ARGS ?=

.PHONY: run test

run:
	$(PYTHON) -m app.main $(ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -v
```

Set `requirements.txt` to:

```text
torch>=2.2,<3
torchvision>=0.17,<1
```

Document setup (`pip install -r requirements.txt`), `make run`, `make test`, CIFAR-10 download location, and every CLI option in the README.

- [ ] **Step 5: Verify the integrated sample**

Run: `cd ml_sample/mrl-image && python -m unittest discover -s tests -v`

Expected: all tests pass without downloading CIFAR-10.

Run: `cd ml_sample/mrl-image && make run ARGS="--epochs 1 --batch-size 64 --embedding-dim 16 --dimensions 4,8,16"`

Expected: CIFAR-10 downloads on first run, followed by an epoch loss and one `accuracy` line for dimensions 4, 8, and 16.

- [ ] **Step 6: Validate the final change set**

Run: `cd /Users/wararaki/workspace/scrapbox9 && git diff --check && git status --short`

Expected: no whitespace errors; only the approved specification, this plan, and `ml_sample/mrl-image/` files are uncommitted.