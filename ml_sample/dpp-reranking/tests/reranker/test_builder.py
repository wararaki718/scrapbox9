import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.reranker import KernelBuilder


class KernelBuilderTests(unittest.TestCase):
    def test_builds_symmetric_positive_semidefinite_quality_kernel(self) -> None:
        kernel = KernelBuilder().build(
            torch.tensor([1.0, 0.0, -1.0]),
            torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
        )

        self.assertTrue(torch.isfinite(kernel).all())
        self.assertTrue(torch.allclose(kernel, kernel.T, atol=1e-6))
        self.assertGreaterEqual(torch.linalg.eigvalsh(kernel).min().item(), -1e-6)
        self.assertGreater(kernel[0, 0].item(), kernel[2, 2].item())

    def test_rejects_mismatched_candidate_counts(self) -> None:
        with self.assertRaises(ValueError):
            KernelBuilder().build(torch.ones(2), torch.ones(3, 2))

    def test_rejects_invalid_ranks_and_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            KernelBuilder().build(torch.ones(2, 1), torch.ones(2, 2))
        with self.assertRaises(ValueError):
            KernelBuilder().build(torch.tensor([1.0, float("nan")]), torch.ones(2, 2))

    def test_rejects_invalid_quality_scale(self) -> None:
        with self.assertRaises(ValueError):
            KernelBuilder(quality_scale=-1.0)


if __name__ == "__main__":
    unittest.main()