from langchain_core.messages import SystemMessage
from langgraph.types import Command

from app.schemas import PurchaseInformation
from ..models import RefundState, RefundStep
from ..prompts import REFUND_INSTRUCTIONS
from ..router import next_refund_step
from ..utils import merge_purchase_information, normalized_line_ids


def gather_info(model, state: RefundState) -> Command[RefundStep]:
    extractor = model.with_structured_output(PurchaseInformation)
    parsed = PurchaseInformation.model_validate(
        extractor.invoke([SystemMessage(content=REFUND_INSTRUCTIONS), *state.get("messages", [])])
    )
    current_invoice_line_ids = normalized_line_ids(parsed.invoice_line_ids)
    merged_state = merge_purchase_information(state, parsed)
    route = next_refund_step(
        parsed.invoice_id,
        current_invoice_line_ids,
        merged_state.get("first_name"),
        merged_state.get("last_name"),
        merged_state.get("phone"),
    )
    return Command(update=merged_state, goto=route)
