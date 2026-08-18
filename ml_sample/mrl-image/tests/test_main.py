import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import ANY, patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from app.train import Trainer


class MainTests(unittest.TestCase):
    @patch("app.main.Trainer", wraps=Trainer)
    @patch("app.main.create_dataloaders")
    def test_main_trains_and_reports_accuracy_for_each_dimension(self, create_dataloaders, trainer_class):
        from app.main import main

        images = torch.randn(4, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 3])
        dataset = TensorDataset(images, labels)
        create_dataloaders.return_value = (
            DataLoader(dataset, batch_size=2),
            DataLoader(dataset, batch_size=2),
        )

        output = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "--epochs",
                    "1",
                    "--embedding-dim",
                    "8",
                    "--dimensions",
                    "4,8",
                    "--device",
                    "cpu",
                ],
            ),
            redirect_stdout(output),
        ):
            main()

        result = output.getvalue()
        self.assertIn("epoch=1 loss=", result)
        self.assertIn("dimension=4 accuracy=", result)
        self.assertIn("dimension=8 accuracy=", result)
        create_dataloaders.assert_called_once_with(
            Path("data"),
            batch_size=64,
            train_samples=1000,
            test_samples=200,
            seed=42,
        )
        trainer_class.assert_called_once_with(ANY, torch.device("cpu"), 1e-3, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
