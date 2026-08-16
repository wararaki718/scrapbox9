import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.loss import matryoshka_infonce_loss


class LossTests(unittest.TestCase):
    def test_loss_backpropagates(self) -> None:
        queries = torch.randn(3, 8, requires_grad=True)
        positives = torch.randn(3, 8, requires_grad=True)
        loss = matryoshka_infonce_loss(queries, positives, [2, 4, 8], 0.1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(queries.grad)
