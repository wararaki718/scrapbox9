import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.schemas import Recommendation
from app.utils import show


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