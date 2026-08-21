from langchain_ollama import ChatOllama


def create_model() -> ChatOllama:
    return ChatOllama(model="qwen3:1.7b", temperature=0)