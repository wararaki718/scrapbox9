import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.main import main


class MainTests(unittest.TestCase):
    def test_main_runs_training_and_prints_both_rankings(self) -> None:
        output = StringIO()
        with redirect_stdout(output):
            main()

        text = output.getvalue()
        self.assertIn("Relevance ranking", text)
        self.assertIn("DPP ranking", text)
        self.assertEqual(text.count("rank  item  category  score"), 2)