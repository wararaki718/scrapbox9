from langchain_core.runnables import RunnableConfig

from app.schemas import AgentState
from ..refund_adapter import parent_to_refund_state, refund_to_parent_update


def refund_agent(graph, state: AgentState, config: RunnableConfig) -> AgentState:
    result = graph.invoke(parent_to_refund_state(state), config=config)
    return refund_to_parent_update(result)