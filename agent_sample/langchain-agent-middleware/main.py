from langchain_ollama import ChatOllama
from deepagents.backends import StateBackend
from deepagents.middleware import FilesystemMiddleware
from deepagents.middleware.subagents import SubAgentMiddleware
from langchain.agents.middleware import TodoListMiddleware

from agent import OllamaAgent
from builder import HumanPromptBuilder, SystemPromptBuilder
from harness import AgentHarness
from tools import search


def main() -> None:
    # init model
    model_name: str = "qwen2.5:0.5b"
    model = ChatOllama(model=model_name)

    # build prompt
    human_prompt_builder = HumanPromptBuilder()
    human_prompt = human_prompt_builder.build()
    system_prompt_builder = SystemPromptBuilder()
    system_prompt = system_prompt_builder.build()

    # middleware
    backend = StateBackend()
    middleware = [
        FilesystemMiddleware(backend=backend),
        TodoListMiddleware(),
        SubAgentMiddleware(
            backend=backend,
            subagents=[
                {
                    "name": "researcher",
                    "description": "Searches and returns a structured summary.",
                    "system_prompt": system_prompt,
                    "tools": [search],
                    "model": model,
                    "middleware": [],
                },
            ],
        ),
    ]
    
    # define harness
    harness = AgentHarness(
        tools=[search],
        middleware=middleware,
    )

    # define agent
    agent = OllamaAgent(
        model=model,
        harness=harness,
    )
    response = agent.ask([human_prompt])
    print("Agent response:")
    print(response[0]["text"])
    print()

    print("DONE")


if __name__ == "__main__":
    
    main()
