from langgraph.types import Command

from app.schemas import AgentState
from ..router import RouteName, classify_intent


def intent_classifier(classifier_model, state: AgentState) -> Command[RouteName]:
    return classify_intent(classifier_model, state)