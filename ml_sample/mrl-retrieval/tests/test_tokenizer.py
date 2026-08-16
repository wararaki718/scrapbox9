import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tokenizer import Tokenizer


class TokenizerTests(unittest.TestCase):
    def test_fit_and_encode_builds_vocabulary_and_pads(self) -> None:
        tokenizer = Tokenizer().fit(["猫"])

        token_ids, mask = tokenizer.encode("犬", 4)

        self.assertEqual(token_ids[0].item(), tokenizer.unknown_id)
        self.assertTrue(torch.equal(mask, torch.tensor([1, 0, 0, 0])))


if __name__ == "__main__":
    unittest.main()