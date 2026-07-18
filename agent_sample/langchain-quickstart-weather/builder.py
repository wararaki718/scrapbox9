from langchain_core.messages import HumanMessage, SystemMessage


class PromptBuilder:
    def build(self, content: str) -> list[HumanMessage | SystemMessage]:
        return [
            SystemMessage("You are a helpful assistant"),
            HumanMessage(content=content),
        ]
