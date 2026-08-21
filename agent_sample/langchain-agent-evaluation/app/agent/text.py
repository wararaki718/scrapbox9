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
        content = content.strip()
        return content or None

    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text = block
        elif isinstance(block, dict):
            text = block.get("text")
        else:
            text = getattr(block, "text", None)

        if isinstance(text, str):
            text = text.strip()
            if text:
                text_parts.append(text)

    return "\n".join(text_parts) if text_parts else None
