import sys
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.agent.refund_adapter import refund_to_parent_update


def test_refund_to_parent_update_preserves_followup_as_history() -> None:
    followup = "Previewed a refund total of $1.23."

    result = refund_to_parent_update({"followup": followup, "invoice_id": 42})

    assert result["followup"] == followup
    assert result["invoice_id"] == 42
    assert any(isinstance(message, AIMessage) and message.content == followup for message in result["messages"])


def test_refund_to_parent_update_does_not_add_empty_followup_to_history() -> None:
    result = refund_to_parent_update({"followup": "", "invoice_id": 42})

    assert result == {"followup": "", "invoice_id": 42}
    assert "messages" not in result
