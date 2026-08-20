from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from refund import create_refund_graph
from schemas import AgentState, UserIntent
from tools import create_catalog_tools

IntentName = Literal["refund", "question_answering"]
RouteName = Literal["refund_agent", "question_answering_agent"]

ROUTER_SYSTEM_PROMPT = """You are the top-level router for a music store support agent.
Choose exactly one intent for the user's latest request:
- refund: the customer wants a refund, return, reversal, repayment, purchase lookup for refund handling, or any help that should continue in the refund workflow.
- question_answering: the customer is asking about the music catalog, such as whether a track, album, or artist exists.

Route to refund when the user mentions refunds even if the request is missing purchase identifiers or customer details.
Route to question_answering only for catalog lookup questions that do not ask for refunds.
Return structured output only."""

QUESTION_ANSWERING_SYSTEM_PROMPT = """You are a music store catalog assistant.
Answer questions about tracks, albums, and artists using the catalog lookup tools.
If a lookup returns no matches, say so clearly and do not invent catalog entries."""


def create_model() -> ChatOllama:
    return ChatOllama(model="qwen3:1.7b", temperature=0)


def normalize_route(intent: IntentName) -> RouteName:
    return {
        "refund": "refund_agent",
        "question_answering": "question_answering_agent",
    }[intent]


def compile_followup(state: AgentState) -> AgentState:
    followup = _normalized_text(state.get("followup"))
    if followup is None:
        followup = _message_content_to_text(_last_message(state.get("messages", [])))
    return {"followup": followup}


def create_support_graph(database: Path):
    database_path = Path(database)
    model = create_model()
    intent_classifier_model = model.with_structured_output(UserIntent)
    question_answering_graph = create_agent(
        model,
        tools=create_catalog_tools(database_path),
        system_prompt=QUESTION_ANSWERING_SYSTEM_PROMPT,
        name="question_answering_agent",
    )
    refund_graph = create_refund_graph(model, database_path)

    def intent_classifier(state: AgentState) -> Command[RouteName]:
        intent = UserIntent.model_validate(
            intent_classifier_model.invoke([SystemMessage(content=ROUTER_SYSTEM_PROMPT), *state.get("messages", [])])
        )
        route = normalize_route(intent.intent)
        return Command(update={"route": route}, goto=route)

    def question_answering_agent(state: AgentState, config: RunnableConfig) -> AgentState:
        result = question_answering_graph.invoke({"messages": state.get("messages", [])}, config=config)
        last_message = _last_message(result.get("messages"))
        return {"messages": [last_message]} if last_message is not None else {}

    def refund_agent(state: AgentState, config: RunnableConfig) -> AgentState:
        result = refund_graph.invoke(_parent_to_refund_state(state), config=config)
        return _refund_to_parent_update(result)

    graph = StateGraph(AgentState)
    graph.add_node("intent_classifier", intent_classifier, destinations=("refund_agent", "question_answering_agent"))
    graph.add_node("question_answering_agent", question_answering_agent)
    graph.add_node("refund_agent", refund_agent)
    graph.add_node("compile_followup", compile_followup)
    graph.add_edge(START, "intent_classifier")
    graph.add_edge("question_answering_agent", "compile_followup")
    graph.add_edge("refund_agent", "compile_followup")
    graph.add_edge("compile_followup", END)
    return graph.compile(name="support_graph")


class _RefundState(TypedDict, total=False):
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


def _parent_to_refund_state(state: AgentState) -> _RefundState:
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


def _refund_to_parent_update(result: _RefundState) -> AgentState:
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
    followup = _normalized_text(result.get("followup"))
    if followup is not None:
        update["messages"] = [AIMessage(content=followup)]
    return update


def _last_message(messages: object) -> AnyMessage | None:
    if not isinstance(messages, list) or not messages:
        return None
    last_message = messages[-1]
    return last_message if hasattr(last_message, "content") else None


def _message_content_to_text(message: AnyMessage | None) -> str | None:
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return _normalized_text(content)
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for block in content:
        text = _content_block_to_text(block)
        if text is not None:
            text_parts.append(text)
    if not text_parts:
        return None
    return "\n".join(text_parts)


def _content_block_to_text(block: object) -> str | None:
    if isinstance(block, str):
        return _normalized_text(block)
    if isinstance(block, dict):
        text = block.get("text")
        return _normalized_text(text) if isinstance(text, str) else None
    text = getattr(block, "text", None)
    return _normalized_text(text) if isinstance(text, str) else None


def _normalized_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
