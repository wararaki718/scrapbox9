from dataclasses import dataclass
from typing import Callable, Sequence, cast

from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware import TodoListMiddleware


@dataclass
class AgentHarness:
    tools: Sequence[Callable[[str], str]] | None = None
    middleware: Sequence[FilesystemMiddleware | SubAgentMiddleware | TodoListMiddleware] | None = None

    def to_dict(self) -> dict[str, Sequence[Callable[[str], str]] | Sequence[FilesystemMiddleware | SubAgentMiddleware | TodoListMiddleware] | None]:
        return cast(
            dict[str, Sequence[Callable[[str], str]] | Sequence[FilesystemMiddleware | SubAgentMiddleware | TodoListMiddleware] | None],
            {
                "tools": self.tools,
                "middleware": self.middleware,
            },
        )
