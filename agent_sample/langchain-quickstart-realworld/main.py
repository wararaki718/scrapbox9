from langchain_ollama import ChatOllama

from agent import OllamaAgent, OllamaDeepAgent
from builder import HumanPromptBuilder, SystemPromptBuilder
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

    # define agent
    agent = OllamaAgent(
        model=model,
        system_prompt=system_prompt,
        tools=[fetch_text, count_lines_containing, find_first_line],
    )
    agent_response = agent.ask([human_prompt])
    print("Agent response:")
    print(agent_response)
    print()

    deep_agent = OllamaDeepAgent(
        model=model,
        system_prompt=system_prompt,
        tools=[fetch_text, count_lines_containing, find_first_line],
    )
    deep_agent_response = deep_agent.ask([human_prompt])
    print("Deep agent response:")
    print(deep_agent_response)
    print()

    print("DONE")


if __name__ == "__main__":
    
    main()
