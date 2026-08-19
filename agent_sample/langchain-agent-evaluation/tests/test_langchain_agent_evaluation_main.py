import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main import build_parser


def test_parser_defaults_to_safe_single_question_mode() -> None:
    args = build_parser().parse_args([])

    assert args.question is None
    assert args.evaluate is False
    assert args.database == 'chinook.db'
