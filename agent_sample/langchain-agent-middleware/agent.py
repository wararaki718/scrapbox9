from typing import cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from harness import AgentHarness


class OllamaAgent:
    def __init__(
        self,
        model: ChatOllama,
        harness: AgentHarness | None = None,
    ) -> None:
        if harness is None:
            harness = AgentHarness()

        self._agent = create_agent(
            model=model,
            checkpointer=InMemorySaver(),
            **harness.to_dict(),
        )

    def ask(self, prompt: list[HumanMessage]) -> list[dict[str, str]]:
        response = self._agent.invoke(
            {"messages": prompt},
            config={"configurable": {"thread_id": "ollama-deep-agent-id"}},
        )
        return cast(list[dict[str, str]], response["messages"][-1].content_blocks)

