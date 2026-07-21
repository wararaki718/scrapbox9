from dataclasses import dataclass
from typing import Callable, Sequence, cast

from langchain_core.messages import SystemMessage
from langchain.agents.structured_output import ToolStrategy

from schema import Answer


@dataclass
class AgentHarness:
    system_prompt: str | SystemMessage | None = None
    tools: Sequence[Callable[[str], str]] | None = None
    response_format: type[Answer] | None = None

    def to_dict(self) -> dict[str, str | SystemMessage | Sequence[Callable[[str], str]] | ToolStrategy | None]:
        return cast(
            dict[str, str | SystemMessage | Sequence[Callable[[str], str]] | ToolStrategy | None],
            {
                "system_prompt": self.system_prompt,
                "tools": self.tools,
                "response_format": ToolStrategy(self.response_format),
            },
        )
