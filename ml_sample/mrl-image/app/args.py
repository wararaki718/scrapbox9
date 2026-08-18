import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import torch


def _positive_int(value: str) -> int:
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed_value


def _positive_float(value: str) -> float:
    parsed_value = float(value)
    if not math.isfinite(parsed_value) or parsed_value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed_value


def _device(value: str) -> str:
    try:
        torch.device(value)
    except RuntimeError as error:
        raise argparse.ArgumentTypeError(f"invalid device: {value}") from error
    return value


def _parse_dimensions(value: str, embedding_dim: int, parser: argparse.ArgumentParser) -> list[int]:
    try:
        dimensions = [int(component) for component in value.split(",")]
    except ValueError:
        parser.error("dimensions must be comma-separated integers")

    if not dimensions:
        parser.error("dimensions must not be empty")
    if any(dimension <= 0 for dimension in dimensions):
        parser.error("dimensions must be positive")
    if any(dimension > embedding_dim for dimension in dimensions):
        parser.error("dimensions must not exceed embedding-dim")
    if len(set(dimensions)) != len(dimensions):
        parser.error("dimensions must not contain duplicates")

    return dimensions


def _parse_loss_weights(
    value: str | None, dimensions: Sequence[int], parser: argparse.ArgumentParser
) -> list[float]:
    if value is None:
        return [1.0] * len(dimensions)
    if not value:
        parser.error("loss-weights must not be empty")

    try:
        loss_weights = [float(component) for component in value.split(",")]
    except ValueError:
        parser.error("loss-weights must be comma-separated numbers")

    if len(loss_weights) != len(dimensions):
        parser.error("loss-weights must contain one value per dimension")
    if any(not math.isfinite(weight) or weight < 0 for weight in loss_weights):
        parser.error("loss-weights must be non-negative finite numbers")

    return loss_weights


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=_positive_int, default=2)
    parser.add_argument("--batch-size", type=_positive_int, default=64)
    parser.add_argument("--train-samples", type=_positive_int, default=1000)
    parser.add_argument("--test-samples", type=_positive_int, default=200)
    parser.add_argument("--embedding-dim", type=_positive_int, default=64)
    parser.add_argument("--dimensions", default="8,16,32,64")
    parser.add_argument("--loss-weights")
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=_device, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args(argv)
    args.dimensions = _parse_dimensions(args.dimensions, args.embedding_dim, parser)
    args.loss_weights = _parse_loss_weights(args.loss_weights, args.dimensions, parser)
    return args