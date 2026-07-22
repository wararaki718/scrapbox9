from langchain_core.messages import HumanMessage, SystemMessage


class HumanPromptBuilder:
    def build(self) -> HumanMessage:
        content = """
        You are an AI agent that can use tools to answer questions.
        When you receive a question, you should first use the search tool to research the question.
        """
        return HumanMessage(content=content)


class SystemPromptBuilder:
    def build(self) -> SystemMessage:
        content = "Use the search tool to research the question and summarize key points."
        return SystemMessage(content=content)
