from pathlib import Path
from typing import TypeVar

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


T = TypeVar("T")


def _sample_dataset(dataset: Dataset[T], sample_count: int, seed: int) -> Subset[T]:
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
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if train_samples <= 0:
        raise ValueError("train_samples must be positive")
    if test_samples <= 0:
        raise ValueError("test_samples must be positive")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616),
            ),
        ]
    )
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    if train_samples > len(train_dataset):
        raise ValueError("train_samples must not exceed dataset size")

    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    if test_samples > len(test_dataset):
        raise ValueError("test_samples must not exceed dataset size")

    train_subset = _sample_dataset(train_dataset, train_samples, seed)
    test_subset = _sample_dataset(test_dataset, test_samples, seed + 1)
    train_generator = torch.Generator().manual_seed(seed)

    return (
        DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=train_generator),
        DataLoader(test_subset, batch_size=batch_size, shuffle=False),
    )