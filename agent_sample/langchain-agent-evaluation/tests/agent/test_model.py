import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.model import create_model


def test_create_model_uses_fixed_ollama_configuration() -> None:
    model = create_model()

    assert model.model == "qwen3:1.7b"
    assert model.temperature == 0.0
