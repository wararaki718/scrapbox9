import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess import CharacterTokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def test_encode_pads_and_handles_unknown_characters(self) -> None:
        tokenizer = CharacterTokenizer().fit(["猫"])

        token_ids, mask = tokenizer.encode("犬", 4)

        self.assertEqual(token_ids.shape, (4,))
        self.assertEqual(token_ids[0].item(), tokenizer.unknown_id)
        self.assertTrue(torch.equal(mask, torch.tensor([1, 0, 0, 0])))


if __name__ == "__main__":
    unittest.main()