import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--dimensions", default="8,16,32,64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)

    # parse the arguments
    args = parser.parse_args(argv)

    # check
    try:
        args.dimensions = [int(value) for value in args.dimensions.split(",") if value]
    except ValueError:
        parser.error("dimensions must be comma-separated integers")

    if not args.dimensions or max(args.dimensions) > args.embedding_dim or min(args.dimensions) <= 0:
        parser.error("dimensions must be positive and no greater than embedding-dim")

    return args
