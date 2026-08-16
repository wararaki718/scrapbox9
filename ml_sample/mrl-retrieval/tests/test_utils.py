import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app.utils as utils
from app.utils import DEFAULT_PAIRS, load_pairs


class UtilsTests(unittest.TestCase):
    def test_returns_default_pairs_without_a_path(self) -> None:
        self.assertEqual(load_pairs(None), DEFAULT_PAIRS)

    def test_loads_jsonl_pair(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as file:
            file.write(json.dumps({"query": "質問", "positive": "回答"}) + "\n")
            file.flush()
            self.assertEqual(load_pairs(Path(file.name)), [("質問", "回答")])

    def test_does_not_wrap_dataloader(self) -> None:
        self.assertFalse(hasattr(utils, "create_dataloader"))

    def test_does_not_define_parse_args(self) -> None:
        self.assertFalse(hasattr(utils, "parse_args"))

