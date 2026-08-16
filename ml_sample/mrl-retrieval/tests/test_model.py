import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.model import MatryoshkaEncoder


class ModelTests(unittest.TestCase):
    def test_normalized_prefix(self) -> None:
        model = MatryoshkaEncoder(12, 8, 2, 1, 4)
        self.assertFalse(hasattr(model, "embedding_dim"))
        self.assertTrue(hasattr(model, "_embedding_dim"))
        embedding = model(torch.tensor([[1, 2, 0, 0]]), torch.tensor([[1, 1, 0, 0]]), 4)
        self.assertEqual(embedding.shape, (1, 4))
        self.assertTrue(torch.allclose(embedding.norm(dim=1), torch.ones(1), atol=1e-6))
