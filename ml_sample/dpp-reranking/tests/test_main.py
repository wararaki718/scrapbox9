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

        sections = output.getvalue().strip().split("\n\n")
        self.assertEqual(len(sections), 2)
        for section, title in zip(
            sections, ("Relevance ranking", "DPP ranking"), strict=True
        ):
            lines = section.splitlines()
            self.assertEqual(lines[0], title)
            self.assertEqual(lines[1], "rank  item  category  score")
            self.assertEqual(len(lines[2:]), 20)