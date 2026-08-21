from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


RefundStep = Literal["refund", "lookup", "respond"]


class RefundState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
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
