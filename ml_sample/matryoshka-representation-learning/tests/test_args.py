import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from args import parse_args


class ArgsTests(unittest.TestCase):
    def test_parse_args_returns_valid_embedding_dimensions(self) -> None:
        args = parse_args(["--embedding-dim", "8", "--dimensions", "4,8"])

        self.assertEqual(args.embedding_dim, 8)
        self.assertEqual(args.dimensions, [4, 8])


if __name__ == "__main__":
    unittest.main()