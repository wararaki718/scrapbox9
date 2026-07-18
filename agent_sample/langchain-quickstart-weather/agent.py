from typing import Callable, Sequence

from langchain_ollama import ChatOllama


class OllamaAgent:
    def __init__(
        self,
        model_name: str = "qwen2.5:0.5b",
        tools: Sequence[Callable[[str], str]] | None = None,
    ) -> None:
        self._model = ChatOllama(model=model_name)
        if tools is not None:
            self._model.bind_tools(tools)

    def ask(self, prompt: list[str]) -> str:
        response = self._model.invoke(prompt)
        return str(getattr(response, "content", response))
