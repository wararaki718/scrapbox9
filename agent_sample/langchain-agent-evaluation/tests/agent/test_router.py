import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.router import normalize_route


def test_normalize_route_maps_supported_intents() -> None:
    assert normalize_route("refund") == "refund_agent"
    assert normalize_route("question_answering") == "question_answering_agent"
