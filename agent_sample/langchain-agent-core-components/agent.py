from typing import cast

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from harness import AgentHarness
from schema import Answer


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

    def ask(self, prompt: list[HumanMessage]) -> Answer:
        response = self._agent.invoke(
            {"messages": prompt},
            config={"configurable": {"thread_id": "ollama-agent-id"}},
        )

        if "structured_response" not in response:
            last_message = response["messages"][-1]
            raise RuntimeError(
                "The model did not return an Answer structured response. "
                f"Last message: {last_message!r}"
            )

        return cast(Answer, response["structured_response"])
