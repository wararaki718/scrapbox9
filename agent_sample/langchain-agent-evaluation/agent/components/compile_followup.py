from schemas import AgentState
from ..text import last_message, message_content_to_text, normalized_text


def compile_followup(state: AgentState) -> AgentState:
    followup = normalized_text(state.get("followup"))
    if followup is None:
        followup = message_content_to_text(last_message(state.get("messages", [])))
    return {"followup": followup}