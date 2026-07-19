from typing import Callable, Sequence

from langchain.agents import create_agent
from langchain_ollama import ChatOllama


class OllamaAgent:
    def __init__(
        self,
        model_name: str = "qwen2.5:0.5b",
        tools: Sequence[Callable[[str], str]] | None = None,
    ) -> None:
        model = ChatOllama(model=model_name)
        if tools is None:
            tools = []
        self._agent = create_agent(model, tools=tools)

    def ask(self, prompt: list[str]) -> str:
        response = self._agent.invoke({"messages": prompt})
        return str(response["messages"][-1].content)
