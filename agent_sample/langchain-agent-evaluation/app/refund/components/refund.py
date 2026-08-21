from pathlib import Path

from langchain_core.runnables import RunnableConfig

from ..models import RefundState
from ..utils import configurable_value


def refund(database: Path, state: RefundState, config: RunnableConfig | None, apply_refund):
    mock = configurable_value(config, "env") != "prod"
    total = apply_refund(
        database,
        state.get("invoice_id"),
        state.get("invoice_line_ids"),
        mock=mock,
    )
    if mock:
        followup = f"Previewed a refund total of ${total:.2f}. No database changes were made because env is not prod."
    else:
        followup = f"Refunded ${total:.2f} successfully."
    return {"followup": followup}
