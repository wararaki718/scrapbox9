from typing import TypedDict

from langchain_core.messages import AIMessage, AnyMessage

from schemas import AgentState
from .text import normalized_text


class RefundState(TypedDict, total=False):
    messages: list[AnyMessage]
    followup: str | None
    invoice_id: int | None
    invoice_line_ids: list[int] | None
    first_name: str | None
    last_name: str | None
    phone: str | None
    track_name: str | None
    album_title: str | None
    artist_name: str | None
    purchase_date_iso_8601: str | None


def parent_to_refund_state(state: AgentState) -> RefundState:
    return {
        "messages": state.get("messages", []),
        "followup": state.get("followup"),
        "invoice_id": state.get("invoice_id"),
        "invoice_line_ids": state.get("invoice_line_ids"),
        "first_name": state.get("customer_first_name"),
        "last_name": state.get("customer_last_name"),
        "phone": state.get("customer_phone"),
        "track_name": state.get("track_name"),
        "album_title": state.get("album_title"),
        "artist_name": state.get("artist_name"),
        "purchase_date_iso_8601": state.get("purchase_date_iso_8601"),
    }


def refund_to_parent_update(result: RefundState) -> AgentState:
    update: AgentState = {}
    for parent_key, child_key in (
        ("followup", "followup"),
        ("invoice_id", "invoice_id"),
        ("invoice_line_ids", "invoice_line_ids"),
        ("track_name", "track_name"),
        ("album_title", "album_title"),
        ("artist_name", "artist_name"),
        ("purchase_date_iso_8601", "purchase_date_iso_8601"),
        ("customer_first_name", "first_name"),
        ("customer_last_name", "last_name"),
        ("customer_phone", "phone"),
    ):
        if child_key in result:
            update[parent_key] = result[child_key]  # type: ignore[literal-required]
    followup = normalized_text(result.get("followup"))
    if followup is not None:
        update["messages"] = [AIMessage(content=followup)]
    return update