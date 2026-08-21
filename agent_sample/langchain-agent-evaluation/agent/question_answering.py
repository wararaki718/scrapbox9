from langchain_core.runnables import RunnableConfig

from schemas import AgentState
from .text import last_message

QUESTION_ANSWERING_SYSTEM_PROMPT = """You are a music store catalog assistant.
Answer questions about tracks, albums, and artists using the catalog lookup tools.
If a lookup returns no matches, say so clearly and do not invent catalog entries."""


def run_question_answering(graph, state: AgentState, config: RunnableConfig) -> AgentState:
    result = graph.invoke({"messages": state.get("messages", [])}, config=config)
    message = last_message(result.get("messages"))
    return {"messages": [message]} if message is not None else {}