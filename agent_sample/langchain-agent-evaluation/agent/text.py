from langchain_core.messages import AnyMessage


def last_message(messages: object) -> AnyMessage | None:
    if not isinstance(messages, list) or not messages:
        return None
    message = messages[-1]
    return message if hasattr(message, "content") else None


def message_content_to_text(message: AnyMessage | None) -> str | None:
    if message is None:
        return None
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return normalized_text(content)
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for block in content:
        text = content_block_to_text(block)
        if text is not None:
            text_parts.append(text)
    return "\n".join(text_parts) if text_parts else None


def content_block_to_text(block: object) -> str | None:
    if isinstance(block, str):
        return normalized_text(block)
    if isinstance(block, dict):
        text = block.get("text")
        return normalized_text(text) if isinstance(text, str) else None
    text = getattr(block, "text", None)
    return normalized_text(text) if isinstance(text, str) else None


def normalized_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None