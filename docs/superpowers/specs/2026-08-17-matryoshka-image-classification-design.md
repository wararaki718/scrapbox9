# Matryoshka Image Classification Design

## Goal

Provide a self-contained PyTorch example of Matryoshka Representation
Learning (MRL) for image classification. The sample trains on CIFAR-10 and
reports test accuracy for several embedding prefix dimensions.

## Scope

The sample is located in `ml_sample/mrl-image/`. It downloads CIFAR-10 through
`torchvision` on first use and stores it in `ml_sample/mrl-image/data/` by
default. The default configuration uses a small CNN and short CPU-oriented
training settings so that the learning workflow is easy to run locally.

The scope excludes pretrained models, distributed training, checkpoints, and
image retrieval. The purpose is to demonstrate that a single model can produce
useful classifications from progressively shorter prefixes of one embedding.

## Architecture

- `app/args.py` exposes `parse_args()`, validates command-line values, and
  converts the comma-separated prefix-dimension option into integers.
- `app/dataset.py` exposes `create_dataloaders()`, which downloads CIFAR-10,
  applies tensor conversion and CIFAR-10 normalization, samples both splits,
  and returns train and test data loaders.
- `app/model.py` defines `MatryoshkaImageClassifier`: a small convolutional
  feature extractor, a projection to the maximum embedding dimension, and one
  linear classifier for each requested prefix dimension.
- `app/loss.py` exposes `matryoshka_cross_entropy()`, which computes the
  relative-importance-weighted sum of cross-entropy losses for all prefix
  logits.
- `app/train.py` defines `Trainer`, which performs one training epoch and
  evaluates accuracy independently for every prefix dimension.
- `app/main.py` defines `main()`, which builds the loaders and model, runs
  training, then prints one test-accuracy line per prefix dimension.

Each application module has one focused responsibility. Tests mirror the
application modules under `tests/`.

## Training Behavior

For an input image, the CNN produces an embedding

$$
z \in \mathbb{R}^{D}
$$

where $D$ is the configured maximum embedding dimension. For every requested
prefix $d$, where $d \leq D$, the corresponding classifier receives

$$
z^{(d)} = z_{1:d}
$$

and produces ten CIFAR-10 class logits. For target class $y$, the training
loss is

$$
L = \sum_{d \in \mathcal{D}}
  c_d \cdot \operatorname{CrossEntropy}(h_d(z^{(d)}), y)
$$

where $\mathcal{D}$ is the selected set of prefix dimensions and $c_d \geq 0$
is that prefix's relative importance. The default uses $c_d = 1$ for every
prefix, matching the standard MRL setting in Appendix A of the paper. This
optimizes all prefixes jointly, so short embeddings remain independently usable.

Evaluation computes accuracy for every prefix classifier on the CIFAR-10 test
set and prints results such as `dimension=16 accuracy=0.4200`.

## CLI And Validation

`app/main.py` supports data directory, epoch count, batch size, embedding
dimension, prefix dimensions, learning rate, random seed, device, and train
and test sample-count options. The default sample counts are 1,000 training
images and 200 test images, making the default run suitable for quick local
validation.

The optional comma-separated `--loss-weights` setting assigns relative
importance to the configured prefix dimensions in order. When omitted, all
weights are `1.0`. It must contain exactly one finite, non-negative value per
prefix dimension. For example, `--dimensions 8,16,32,64 --loss-weights
2,1,1,1` doubles the relative contribution of the 8-dimensional classifier.

`create_dataloaders()` uses a locally seeded `torch.Generator` and a random
permutation to select each split without replacement. The seed is supplied by
the CLI, so the same configuration selects the same examples on every run.
The training and test sample counts are positive and cannot exceed their
respective CIFAR-10 split sizes. To train on all CIFAR-10 images, users set
`--train-samples 50000 --test-samples 10000`.

The program rejects non-positive batch sizes, epochs, embedding dimensions,
learning rates, and sample counts; empty or duplicate prefix dimensions; and
dimensions that exceed the embedding dimension; and malformed, mismatched,
negative, or non-finite loss weights.

The default prefix dimensions are `8,16,32,64` with an embedding dimension of
`64`. The default device is automatically selected as CUDA when available and
CPU otherwise; `--device` overrides it.

`Makefile` provides `make run` for the sample and `make test` for its unit
tests. `requirements.txt` lists PyTorch and torchvision. The README includes
setup, execution, and CLI examples.

## Tests

Tests use Python's `unittest` and avoid downloading CIFAR-10 by injecting
small in-memory tensor datasets wherever data loading is not under test.

- argument parsing and invalid-value rejection;
- data loader output shape and class labels, with CIFAR-10 construction mocked;
- embedding and prefix-logit shapes;
- finite MRL loss and backpropagation through every classifier;
- relative-importance parsing, validation, and weighted-loss calculation;
- optimizer-driven parameter updates and per-prefix accuracy accounting;
- a lightweight `main()` integration run using patched data loaders.
- reproducible train/test sampling, sample-count validation, and sampler-size
  propagation into data loaders.