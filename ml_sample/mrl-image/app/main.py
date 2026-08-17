import torch

from app.args import parse_args
from app.dataset import create_dataloaders
from app.model import MatryoshkaImageClassifier
from app.train import Trainer


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_loader, test_loader = create_dataloaders(args.data_dir, args.batch_size)
    model = MatryoshkaImageClassifier(args.embedding_dim, args.dimensions)
    trainer = Trainer(model, torch.device(args.device), args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        loss = trainer.train_epoch(train_loader)
        print(f"epoch={epoch} loss={loss:.4f}")

    accuracy_by_dimension = trainer.evaluate(test_loader)
    for dimension in args.dimensions:
        print(f"dimension={dimension} accuracy={accuracy_by_dimension[dimension]:.4f}")


if __name__ == "__main__":
    main()
