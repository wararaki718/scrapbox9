import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluate import EvaluationResult
from app.schemas import Recommendation
from app.utils import show, show_evaluation


class UtilsTests(unittest.TestCase):
    def test_show_formats_both_lists_to_supplied_stream(self) -> None:
        output = StringIO()
        show(
            [Recommendation(1, "books", 0.75), Recommendation(2, "music", 0.50)],
            [Recommendation(2, "music", 0.50)],
            stream=output,
        )

        relevance, reranked = output.getvalue().strip().split("\n\n")
        relevance_lines = relevance.splitlines()
        reranked_lines = reranked.splitlines()

        self.assertEqual(relevance_lines[1], "rank  item  category  score   label")
        self.assertTrue(relevance_lines[2].endswith("-"))
        self.assertTrue(relevance_lines[3].endswith("reranked"))
        self.assertTrue(reranked_lines[2].endswith("reranked"))

    def test_show_evaluation_formats_metric_table(self) -> None:
        output = StringIO()
        show_evaluation(
            [
                EvaluationResult(
                    "DPP",
                    (1, 2),
                    ((5, 0.8), (10, 0.7), (20, 0.6)),
                    0.4,
                    3,
                )
            ],
            stream=output,
        )

        text = output.getvalue()
        self.assertIn("Offline evaluation", text)
        self.assertIn("NDCG@5  NDCG@10  NDCG@20  ILD@20  coverage@20", text)
        self.assertIn("DPP", text)
        self.assertIn("0.8000", text)