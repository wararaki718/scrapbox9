from langchain_core.runnables import RunnableConfig

from app.schemas import AgentState
from ..question_answering import run_question_answering


def question_answering_agent(graph, state: AgentState, config: RunnableConfig) -> AgentState:
    return run_question_answering(graph, state, config)