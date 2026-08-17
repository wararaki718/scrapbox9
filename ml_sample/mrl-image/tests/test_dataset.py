import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset import create_dataloaders


class DatasetTests(unittest.TestCase):
    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_returns_expected_batches(self, cifar10: object) -> None:
        train_dataset = TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))
        test_dataset = TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([2, 3]))
        cifar10.side_effect = [train_dataset, test_dataset]  # type: ignore[attr-defined]

        train_loader, test_loader = create_dataloaders(Path("data"), batch_size=2)

        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (2, 3, 32, 32))
        self.assertEqual(labels.shape, (2,))
        self.assertEqual(len(test_loader.dataset), 2)

    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_rejects_non_positive_batch_size_before_dataset_construction(
        self, cifar10: object
    ) -> None:
        with self.assertRaises(ValueError):
            create_dataloaders(Path("data"), batch_size=0)

        cifar10.assert_not_called()  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()