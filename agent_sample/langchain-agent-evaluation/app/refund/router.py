from typing import Literal

from .models import RefundStep
from .utils import normalized_text


def next_refund_step(
    invoice_id: int | None,
    invoice_line_ids: list[int] | None,
    first_name: str | None,
    last_name: str | None,
    phone: str | None,
) -> RefundStep:
    if invoice_id is not None or bool(invoice_line_ids):
        return "refund"
    if all(normalized_text(value) for value in (first_name, last_name, phone)):
        return "lookup"
    return "respond"
