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
        self.assertEqual(len(sections), 3)
        relevance_lines = sections[0].splitlines()
        reranked_lines = sections[1].splitlines()
        evaluation_lines = sections[2].splitlines()

        self.assertEqual(relevance_lines[0], "Relevance ranking")
        self.assertEqual(reranked_lines[0], "DPP ranking")
        self.assertEqual(len(relevance_lines[2:]), 20)
        self.assertEqual(len(reranked_lines[2:]), 10)
        self.assertEqual(
            sum(line.endswith("reranked") for line in relevance_lines[2:]), 10
        )
        self.assertTrue(all(line.endswith("reranked") for line in reranked_lines[2:]))

        relevance_item_ids = {int(line.split()[1]) for line in relevance_lines[2:]}
        reranked_item_ids = {int(line.split()[1]) for line in reranked_lines[2:]}
        self.assertTrue(reranked_item_ids <= relevance_item_ids)
        self.assertEqual(evaluation_lines[0], "Offline evaluation")
        self.assertTrue(any(line.startswith("Reranking 1") for line in evaluation_lines))
        self.assertTrue(any(line.startswith("DPP") for line in evaluation_lines))