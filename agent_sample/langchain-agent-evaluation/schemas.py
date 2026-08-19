from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    followup: str | None
    invoice_id: int | None
    invoice_line_ids: list[int] | None
    customer_first_name: str | None
    customer_last_name: str | None
    customer_phone: str | None
    track_name: str | None
    album_title: str | None
    artist_name: str | None
    purchase_date_iso_8601: str | None
    route: Literal["refund_agent", "question_answering_agent"] | None


class PurchaseInformation(BaseModel):
    followup: str | None = Field(
        default=None,
        description="Follow-up question needed before a purchase lookup or refund can continue.",
    )
    invoice_id: int | None = Field(
        default=None,
        description="Invoice identifier when the user clearly refers to a whole purchase.",
    )
    invoice_line_ids: list[int] | None = Field(
        default=None,
        description="Specific invoice line identifiers when the user refers to item-level refunds.",
    )
    customer_first_name: str | None = Field(
        default=None,
        description="Customer first name extracted from the user's request.",
    )
    customer_last_name: str | None = Field(
        default=None,
        description="Customer last name extracted from the user's request.",
    )
    customer_phone: str | None = Field(
        default=None,
        description="Customer phone number extracted from the user's request.",
    )
    track_name: str | None = Field(
        default=None,
        description="Track title extracted from the user's request.",
    )
    album_title: str | None = Field(
        default=None,
        description="Album title extracted from the user's request.",
    )
    artist_name: str | None = Field(
        default=None,
        description="Artist name extracted from the user's request.",
    )
    purchase_date_iso_8601: str | None = Field(
        default=None,
        description="Purchase date extracted from the user's request in ISO 8601 format.",
    )


class UserIntent(BaseModel):
    intent: Literal["refund", "question_answering"] = Field(
        description="Top-level user intent used to route the request."
    )
