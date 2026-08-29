import sys
import unittest
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data import Recommendation
from app.utils import show


class UtilsTests(unittest.TestCase):
    def test_show_formats_both_lists_to_supplied_stream(self) -> None:
        output = StringIO()
        show(
            [Recommendation(1, "books", 0.75)],
            [Recommendation(2, "music", 0.50)],
            stream=output,
        )

        text = output.getvalue()
        self.assertIn("Relevance ranking", text)
        self.assertIn("DPP ranking", text)
        self.assertIn("books", text)
        self.assertIn("music", text)
        self.assertIn("0.7500", text)