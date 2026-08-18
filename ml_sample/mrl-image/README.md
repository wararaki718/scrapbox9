# Matryoshka Image Classification

Python 3.10 or later is required.

## Setup

```sh
python3.11 -m pip install -r requirements.txt
```

## Run

```sh
make run
make test
```

CIFAR-10 is downloaded automatically to `data/` on the first `make run`. By default, each run uses a reproducibly sampled 1,000 training images and 200 test images based on `--seed`. Training output reports accuracy for each configured embedding prefix dimension.

Available CLI options: `--data-dir`, `--epochs`, `--batch-size`, `--train-samples`, `--test-samples`, `--embedding-dim`, `--dimensions`, `--learning-rate`, `--seed`, and `--device`.

For example, to train with custom prefix dimensions:

```sh
make run ARGS="--embedding-dim 32 --dimensions 8,16,32"
```

To train on the full CIFAR-10 datasets:

```sh
make run ARGS="--train-samples 50000 --test-samples 10000"
```
