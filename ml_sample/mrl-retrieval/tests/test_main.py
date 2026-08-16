import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import main


class MainTests(unittest.TestCase):
    def test_main_reports_dimensions(self) -> None:
        arguments = ["main.py", "--epochs", "1", "--batch-size", "4", "--embedding-dim", "8", "--dimensions", "4,8", "--max-length", "16", "--num-heads", "2", "--num-layers", "1"]
        output = io.StringIO()
        with patch.object(sys, "argv", arguments), redirect_stdout(output):
            result = main.main()
        self.assertIn("dimension=4 Recall@1=", output.getvalue())
        self.assertIn("dimension=8 Recall@1=", output.getvalue())
        self.assertIsNone(result)