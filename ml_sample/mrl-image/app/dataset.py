from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

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
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )