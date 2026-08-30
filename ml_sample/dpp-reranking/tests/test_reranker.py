import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.reranker import DPPReranker, GreedyMapSelector, KernelBuilder


class RerankerModuleTests(unittest.TestCase):
    def test_each_class_is_defined_in_its_own_module(self) -> None:
        self.assertEqual(KernelBuilder.__module__, "app.reranker.builder")
        self.assertEqual(GreedyMapSelector.__module__, "app.reranker.selector")
        self.assertEqual(DPPReranker.__module__, "app.reranker.reranker")


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


class GreedyMapSelectorTests(unittest.TestCase):
    def test_selects_quality_then_complementary_candidate(self) -> None:
        kernel = torch.tensor(
            [[4.0, 3.9, 0.0], [3.9, 4.0, 0.0], [0.0, 0.0, 1.0]]
        )
        self.assertEqual(GreedyMapSelector().select(kernel, 2), [0, 2])

    def test_ties_follow_original_candidate_order(self) -> None:
        self.assertEqual(GreedyMapSelector().select(torch.eye(3), 2), [0, 1])

    def test_rejects_invalid_selection_count(self) -> None:
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.eye(2), 3)

    def test_rejects_non_square_and_non_finite_kernels(self) -> None:
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.ones(2, 3), 1)
        with self.assertRaises(ValueError):
            GreedyMapSelector().select(torch.tensor([[float("nan")]]), 1)


class DPPRerankerTests(unittest.TestCase):
    def test_reranks_candidates_through_builder_and_selector(self) -> None:
        indices = DPPReranker().rerank(
            torch.tensor([1.0, 0.9, 0.1]),
            torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            top_k=2,
        )
        self.assertEqual(indices, [0, 2])