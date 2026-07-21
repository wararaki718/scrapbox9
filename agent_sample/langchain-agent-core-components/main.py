from langchain_ollama import ChatOllama

from agent import OllamaAgent
from builder import HumanPromptBuilder, SystemPromptBuilder
from harness import AgentHarness
from schema import Answer
from tools import fetch_text, count_lines_containing, find_first_line


def main() -> None:
    # init model
    model_name: str = "qwen2.5:0.5b"
    model = ChatOllama(model=model_name)

    # build prompt
    human_prompt_builder = HumanPromptBuilder()
    human_prompt = human_prompt_builder.build()
    system_prompt_builder = SystemPromptBuilder()
    system_prompt = system_prompt_builder.build()
    
    # define harness
    harness = AgentHarness(
        system_prompt=system_prompt,
        tools=[fetch_text, count_lines_containing, find_first_line],
        response_format=Answer,
    )

    # define agent
    agent = OllamaAgent(
        model=model,
        harness=harness,
    )
    agent_response = agent.ask([human_prompt])
    print("Agent response:")
    print(agent_response.model_dump_json(indent=2))
    print()

    print("DONE")


if __name__ == "__main__":
    
    main()
