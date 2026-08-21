from app.schemas import AgentState
from ..text import last_message, message_content_to_text


def compile_followup(state: AgentState) -> AgentState:
    followup = state.get("followup")
    if followup is None:
        followup = message_content_to_text(last_message(state.get("messages", [])))
        if isinstance(followup, str):
            followup = followup.strip() or None
    return {"followup": followup}
