import torch
from torch.utils.data import DataLoader

from args import parse_args
from dataset import PairDataset
from evaluate import evaluate
from model import MatryoshkaEncoder
from preprocess import CharacterTokenizer
from train import Trainer
from utils import (
    DEFAULT_PAIRS,
    load_pairs,
    set_seed,
)


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    pairs = load_pairs(__import__("pathlib").Path(args.data_path)) if args.data_path else DEFAULT_PAIRS
    tokenizer = CharacterTokenizer().fit([text for pair in pairs for text in pair])
    dataset = PairDataset(pairs, tokenizer, args.max_length)
    device = torch.device(args.device)
    model = MatryoshkaEncoder(tokenizer.vocabulary_size, args.embedding_dim, args.num_heads, args.num_layers, args.max_length)

    # train
    trainer = Trainer(model, args.dimensions, 0.1, 1e-3, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(1, args.epochs + 1):
        print(f"epoch={epoch} loss={trainer.train_epoch(loader):.4f}")

    # evaluate
    evaluation = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    evaluate(model, evaluation, args.dimensions, device)


if __name__ == "__main__":
    main()