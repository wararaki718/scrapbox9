import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.dataset import _sample_dataset, create_dataloaders


class DatasetTests(unittest.TestCase):
    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_returns_expected_batches(self, cifar10: object) -> None:
        train_dataset = TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))
        test_dataset = TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([2, 3]))
        cifar10.side_effect = [train_dataset, test_dataset]  # type: ignore[attr-defined]

        train_loader, test_loader = create_dataloaders(
            Path("data"), batch_size=2, train_samples=2, test_samples=2, seed=42
        )

        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (2, 3, 32, 32))
        self.assertEqual(labels.shape, (2,))
        self.assertEqual(len(test_loader.dataset), 2)

    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_samples_deterministically(self, cifar10: object) -> None:
        train_dataset = TensorDataset(torch.rand(10, 3, 32, 32), torch.arange(10))
        test_dataset = TensorDataset(torch.rand(10, 3, 32, 32), torch.arange(10))
        cifar10.side_effect = [train_dataset, test_dataset, train_dataset, test_dataset]  # type: ignore[attr-defined]

        first_train_loader, first_test_loader = create_dataloaders(
            Path("data"), batch_size=2, train_samples=4, test_samples=3, seed=42
        )
        second_train_loader, second_test_loader = create_dataloaders(
            Path("data"), batch_size=2, train_samples=4, test_samples=3, seed=42
        )

        self.assertEqual(len(first_train_loader.dataset), 4)
        self.assertEqual(len(first_test_loader.dataset), 3)
        self.assertEqual(first_train_loader.dataset.indices, second_train_loader.dataset.indices)  # type: ignore[attr-defined]
        self.assertEqual(first_test_loader.dataset.indices, second_test_loader.dataset.indices)  # type: ignore[attr-defined]

    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_shuffles_train_batches_deterministically_without_global_rng_mutation(
        self, cifar10: object
    ) -> None:
        train_dataset = TensorDataset(torch.arange(10), torch.arange(10))
        test_dataset = TensorDataset(torch.arange(10), torch.arange(10))
        cifar10.side_effect = [train_dataset, test_dataset, train_dataset, test_dataset]  # type: ignore[attr-defined]
        rng_state = torch.get_rng_state()

        first_train_loader, _ = create_dataloaders(
            Path("data"), batch_size=2, train_samples=6, test_samples=3, seed=42
        )
        second_train_loader, _ = create_dataloaders(
            Path("data"), batch_size=2, train_samples=6, test_samples=3, seed=42
        )

        first_order = torch.cat([labels for _, labels in first_train_loader]).tolist()
        second_order = torch.cat([labels for _, labels in second_train_loader]).tolist()

        self.assertEqual(first_order, second_order)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))

    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_rejects_train_sample_count_larger_than_dataset_before_test_construction(
        self, cifar10: object
    ) -> None:
        train_dataset = TensorDataset(torch.rand(10, 3, 32, 32), torch.arange(10))
        cifar10.return_value = train_dataset  # type: ignore[attr-defined]

        with self.assertRaisesRegex(ValueError, "train_samples must not exceed dataset size"):
            create_dataloaders(
                Path("data"), batch_size=2, train_samples=11, test_samples=3, seed=42
            )

        self.assertEqual(cifar10.call_count, 1)  # type: ignore[attr-defined]

    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_rejects_test_sample_count_larger_than_dataset(
        self, cifar10: object
    ) -> None:
        train_dataset = TensorDataset(torch.rand(10, 3, 32, 32), torch.arange(10))
        test_dataset = TensorDataset(torch.rand(10, 3, 32, 32), torch.arange(10))
        cifar10.side_effect = [train_dataset, test_dataset]  # type: ignore[attr-defined]

        with self.assertRaisesRegex(ValueError, "test_samples must not exceed dataset size"):
            create_dataloaders(
                Path("data"), batch_size=2, train_samples=3, test_samples=11, seed=42
            )

        self.assertEqual(cifar10.call_count, 2)  # type: ignore[attr-defined]

    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_rejects_zero_train_samples_before_dataset_construction(
        self, cifar10: object
    ) -> None:
        with self.assertRaisesRegex(ValueError, "train_samples must be positive"):
            create_dataloaders(
                Path("data"), batch_size=2, train_samples=0, test_samples=3, seed=42
            )

        cifar10.assert_not_called()  # type: ignore[attr-defined]

    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_rejects_zero_test_samples_before_dataset_construction(
        self, cifar10: object
    ) -> None:
        with self.assertRaisesRegex(ValueError, "test_samples must be positive"):
            create_dataloaders(
                Path("data"), batch_size=2, train_samples=3, test_samples=0, seed=42
            )

        cifar10.assert_not_called()  # type: ignore[attr-defined]

    def test_sample_dataset_selects_different_indices_for_different_seeds(self) -> None:
        dataset = TensorDataset(torch.arange(10))

        first_subset = _sample_dataset(dataset, sample_count=4, seed=42)
        second_subset = _sample_dataset(dataset, sample_count=4, seed=43)

        self.assertNotEqual(first_subset.indices, second_subset.indices)

    @patch("app.dataset.datasets.CIFAR10")
    def test_create_dataloaders_rejects_non_positive_batch_size_before_dataset_construction(
        self, cifar10: object
    ) -> None:
        with self.assertRaises(ValueError):
            create_dataloaders(
                Path("data"), batch_size=0, train_samples=2, test_samples=2, seed=42
            )

        cifar10.assert_not_called()  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()