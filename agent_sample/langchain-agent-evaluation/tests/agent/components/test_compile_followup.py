import sys
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.agent.components.compile_followup import compile_followup


def test_compile_followup_keeps_existing_followup() -> None:
    result = compile_followup(
        {"followup": "Please confirm the invoice number.", "messages": [AIMessage(content="ignored")]}
    )

    assert result == {"followup": "Please confirm the invoice number."}


def test_compile_followup_extracts_text_from_message_content_blocks() -> None:
    result = compile_followup(
        {
            "messages": [
                AIMessage(
                    content=[
                        {"type": "text", "text": "We have Black Dog available."},
                        {"type": "image_url", "image_url": "ignored"},
                        "Anything else I can help with?",
                    ]
                )
            ]
        }
    )

    assert result == {"followup": "We have Black Dog available.\nAnything else I can help with?"}
