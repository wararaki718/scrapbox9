from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from tabulate import tabulate

from database import lookup_purchases, refund as apply_refund
from schemas import PurchaseInformation

REFUND_INSTRUCTIONS = """You extract refund workflow information for a music store support agent.
Identify only details that are explicitly stated or directly implied by the customer.
Prefer invoice_id when the user names a full invoice.
Prefer invoice_line_ids when the user names specific purchased items or selected purchase rows.
Collect customer_first_name, customer_last_name, and customer_phone whenever present.
Collect track_name, album_title, artist_name, and purchase_date_iso_8601 when they help identify a purchase.
Write purchase_date_iso_8601 in ISO 8601 form when you can determine it.
If the request still needs identity details before a purchase lookup can continue, write a short followup question.
Do not invent identifiers, names, phone numbers, products, or dates.
Return structured data only."""

MISSING_IDENTITY_FOLLOWUP = (
    "Please share the customer's first name, last name, and phone number so I can look up the purchase."
)
NO_PURCHASES_FOLLOWUP = (
    "I couldn't find any matching purchases. Please verify the customer's first name, last name, "
    "phone number, and any track, album, artist, or purchase date details."
)


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


def next_refund_step(
    invoice_id: int | None,
    invoice_line_ids: list[int] | None,
    first_name: str | None,
    last_name: str | None,
    phone: str | None,
) -> Literal['refund', 'lookup', 'respond']:
    if invoice_id is not None or bool(invoice_line_ids):
        return 'refund'
    if all(_normalized_text(value) for value in (first_name, last_name, phone)):
        return 'lookup'
    return 'respond'


def create_refund_graph(model: ChatOllama, database: Path):
    database_path = Path(database)

    def gather_info(state: RefundState) -> Command[Literal['refund', 'lookup', 'respond']]:
        extractor = model.with_structured_output(PurchaseInformation)
        parsed = PurchaseInformation.model_validate(
            extractor.invoke([SystemMessage(content=REFUND_INSTRUCTIONS), *state.get('messages', [])])
        )
        current_invoice_line_ids = _normalized_line_ids(parsed.invoice_line_ids)
        merged_state = _merge_purchase_information(state, parsed)
        route = next_refund_step(
            parsed.invoice_id,
            current_invoice_line_ids,
            merged_state.get('first_name'),
            merged_state.get('last_name'),
            merged_state.get('phone'),
        )
        return Command(update=merged_state, goto=route)

    def lookup(state: RefundState) -> RefundState:
        rows = lookup_purchases(
            database_path,
            state.get('first_name') or '',
            state.get('last_name') or '',
            state.get('phone') or '',
            state.get('track_name'),
            state.get('album_title'),
            state.get('artist_name'),
            state.get('purchase_date_iso_8601'),
        )
        if not rows:
            return {'invoice_line_ids': [], 'followup': NO_PURCHASES_FOLLOWUP}
        return {
            'invoice_line_ids': [row['invoice_line_id'] for row in rows],
            'followup': 'I found these matching purchases:\n'
            + tabulate(rows, headers='keys', tablefmt='github', floatfmt='.2f'),
        }

    def refund(state: RefundState, config: RunnableConfig) -> RefundState:
        env = _configurable_value(config, 'env')
        mock = env != 'prod'
        total = apply_refund(
            database_path,
            state.get('invoice_id'),
            state.get('invoice_line_ids'),
            mock=mock,
        )
        if mock:
            followup = (
                f'Previewed a refund total of ${total:.2f}. No database changes were made because env is not prod.'
            )
        else:
            followup = f'Refunded ${total:.2f} successfully.'
        return {'followup': followup}

    def respond(state: RefundState) -> RefundState:
        return {'followup': _normalized_text(state.get('followup')) or MISSING_IDENTITY_FOLLOWUP}

    graph = StateGraph(RefundState)
    graph.add_node('gather_info', gather_info, destinations=('refund', 'lookup', 'respond'))
    graph.add_node('lookup', lookup)
    graph.add_node('refund', refund)
    graph.add_node('respond', respond)
    graph.add_edge(START, 'gather_info')
    graph.add_edge('lookup', END)
    graph.add_edge('refund', END)
    graph.add_edge('respond', END)
    return graph.compile(name='refund_graph')


def _merge_purchase_information(state: RefundState, purchase_information: PurchaseInformation) -> RefundState:
    merged_state: RefundState = dict(state)
    invoice_line_ids = _normalized_line_ids(purchase_information.invoice_line_ids)
    merged_state['followup'] = _normalized_text(purchase_information.followup)
    merged_state.update(_merge_value(merged_state, 'invoice_id', purchase_information.invoice_id))
    merged_state.update(_merge_value(merged_state, 'invoice_line_ids', invoice_line_ids))
    if purchase_information.invoice_id is not None:
        merged_state['invoice_line_ids'] = None
    elif invoice_line_ids:
        merged_state['invoice_id'] = None
    merged_state.update(
        _merge_value(merged_state, 'first_name', _normalized_text(purchase_information.customer_first_name))
    )
    merged_state.update(
        _merge_value(merged_state, 'last_name', _normalized_text(purchase_information.customer_last_name))
    )
    merged_state.update(_merge_value(merged_state, 'phone', _normalized_text(purchase_information.customer_phone)))
    merged_state.update(_merge_value(merged_state, 'track_name', _normalized_text(purchase_information.track_name)))
    merged_state.update(_merge_value(merged_state, 'album_title', _normalized_text(purchase_information.album_title)))
    merged_state.update(_merge_value(merged_state, 'artist_name', _normalized_text(purchase_information.artist_name)))
    merged_state.update(
        _merge_value(
            merged_state,
            'purchase_date_iso_8601',
            _normalized_text(purchase_information.purchase_date_iso_8601),
        )
    )
    return merged_state


def _merge_value(state: RefundState, key: str, value: object) -> RefundState:
    if value is None:
        return {} if key in state else {}
    return {key: value}


def _normalized_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalized_line_ids(value: list[int] | None) -> list[int] | None:
    if value is None:
        return None
    return [int(invoice_line_id) for invoice_line_id in value]


def _configurable_value(config: RunnableConfig | None, key: str) -> object:
    if not config:
        return None
    configurable = config.get('configurable')
    if isinstance(configurable, dict):
        return configurable.get(key)
    return None
