import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langchain-agent-evaluation",
        description="Local-first LangGraph music-store agent evaluation scaffold.",
    )
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--database", default="chinook.db")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    if args.question is None and not args.evaluate:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
