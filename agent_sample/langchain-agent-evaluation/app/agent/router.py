from typing import Literal

from langchain_core.messages import SystemMessage
from langgraph.types import Command

from app.schemas import AgentState, UserIntent

IntentName = Literal["refund", "question_answering"]
RouteName = Literal["refund_agent", "question_answering_agent"]

ROUTER_SYSTEM_PROMPT = """You are the top-level router for a music store support agent.
Choose exactly one intent for the user's latest request:
- refund: the customer wants a refund, return, reversal, repayment, purchase lookup for refund handling, or any help that should continue in the refund workflow.
- question_answering: the customer is asking about the music catalog, such as whether a track, album, or artist exists.

Route to refund when the user mentions refunds even if the request is missing purchase identifiers or customer details.
Route to question_answering only for catalog lookup questions that do not ask for refunds.
Return structured output only."""


def normalize_route(intent: IntentName) -> RouteName:
    return {"refund": "refund_agent", "question_answering": "question_answering_agent"}[intent]


def classify_intent(classifier_model, state: AgentState) -> Command[RouteName]:
    intent = UserIntent.model_validate(
        classifier_model.invoke([SystemMessage(content=ROUTER_SYSTEM_PROMPT), *state.get("messages", [])])
    )
    route = normalize_route(intent.intent)
    return Command(update={"route": route}, goto=route)