from collections.abc import Mapping

from langchain_core.messages import AnyMessage


def normalize_result(result: object) -> dict[str, object]:
    normalized = dict(result) if isinstance(result, Mapping) else {"result": result}
    response = normalized_text(normalized.get("followup"))
    if response is None:
        response = message_content_to_text(last_message(normalized.get("messages")))
    normalized["response"] = response or ""
    return normalized


def response_from_event(event: object) -> str | None:
    for candidate in (event_result(event), event):
        if not isinstance(candidate, Mapping):
            continue
        followup = normalized_text(candidate.get("followup"))
        if followup is not None:
            return followup
        message_text = message_content_to_text(last_message(candidate.get("messages")))
        if message_text is not None:
            return message_text
    return None


def event_result(event: object) -> object:
    if not isinstance(event, Mapping):
        return None
    for container in (mapping_value(event, "payload"), mapping_value(event, "data"), event):
        if isinstance(container, Mapping):
            for key in ("result", "output"):
                if key in container:
                    return container[key]
    return None


def last_message(messages: object) -> AnyMessage | Mapping[str, object] | None:
    if not isinstance(messages, list) or not messages:
        return None
    message = messages[-1]
    if hasattr(message, "content") or isinstance(message, Mapping):
        return message
    return None


def message_content_to_text(message: AnyMessage | Mapping[str, object] | None) -> str | None:
    if message is None:
        return None
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if isinstance(content, str):
        return normalized_text(content)
    if not isinstance(content, list):
        return None
    text_parts: list[str] = []
    for block in content:
        text = block if isinstance(block, str) else block.get("text") if isinstance(block, Mapping) else getattr(block, "text", None)
        if isinstance(text, str):
            text = normalized_text(text)
            if text is not None:
                text_parts.append(text)
    return "\n".join(text_parts) if text_parts else None


def mapping_value(mapping: object, key: str) -> object:
    return mapping.get(key) if isinstance(mapping, Mapping) else None


def normalized_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
