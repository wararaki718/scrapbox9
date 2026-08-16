import sys
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import PairDataset
from model import MatryoshkaEncoder
from preprocess import CharacterTokenizer
from train import Trainer


class TrainerTests(unittest.TestCase):
    def test_train_epoch_updates_parameters(self) -> None:
        pairs = [("東京", "東京の天気"), ("富士山", "富士山の高さ")]
        tokenizer = CharacterTokenizer().fit([text for pair in pairs for text in pair])
        model = MatryoshkaEncoder(tokenizer.vocabulary_size, 8, 2, 1, 8)
        before = next(model.parameters()).detach().clone()
        loader = DataLoader(PairDataset(pairs, tokenizer, 8), batch_size=2, shuffle=False)
        Trainer(model, [4, 8], 0.1, 1e-2, torch.device("cpu")).train_epoch(loader)
        self.assertFalse(torch.equal(before, next(model.parameters()).detach()))
