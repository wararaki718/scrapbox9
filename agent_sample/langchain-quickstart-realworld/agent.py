from typing import Callable, Sequence, cast

from deepagents import create_deep_agent
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver


class OllamaAgent:
    def __init__(
        self,
        model: ChatOllama,
        system_prompt: str | SystemMessage = "You are a helpful assistant",
        tools: Sequence[Callable[[str], str]] | None = None,
    ) -> None:
        
        if tools is None:
            tools = []
        self._agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=InMemorySaver(),
        )

    def ask(self, prompt: list[HumanMessage]) -> list[dict[str, str]]:
        response = self._agent.invoke(
            {"messages": prompt},
            config={"configurable": {"thread_id": "ollama-agent-id"}},
        )
        return cast(list[dict[str, str]], response["messages"][-1].content_blocks)


class OllamaDeepAgent:
    def __init__(
        self,
        model: ChatOllama,
        system_prompt: str | SystemMessage  = "You are a helpful assistant",
        tools: Sequence[Callable[[str], str]] | None = None,
    ) -> None:
        if tools is None:
            tools = []
        self._agent = create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            checkpointer=InMemorySaver(),
        )

    def ask(self, prompt: list[HumanMessage]) -> list[dict[str, str]]:
        response = self._agent.invoke(
            {"messages": prompt},
            config={"configurable": {"thread_id": "ollama-deep-agent-id"}},
        )
        return cast(list[dict[str, str]], response["messages"][-1].content_blocks)
