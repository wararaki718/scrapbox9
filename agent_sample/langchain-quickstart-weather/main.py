from agent import OllamaAgent
from builder import PromptBuilder
from tools import get_weather


def main() -> None:
    # build prompt
    builder = PromptBuilder()
    prompt = builder.build("What's the weather in San Francisco?")

    # ask agent
    agent = OllamaAgent(tools=[get_weather])
    response = agent.ask(prompt)
    print(response)

    print("DONE")


if __name__ == "__main__":
    main()
