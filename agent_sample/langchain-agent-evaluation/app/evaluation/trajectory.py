from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph

from .response import response_from_event, mapping_value


async def run_with_trajectory(graph: CompiledStateGraph, question: str) -> dict[str, object]:
    trajectory: list[str] = []
    response: str | None = None
    seen_tool_batches: set[tuple[str, tuple[str, ...]]] = set()
    state = {"messages": [{"role": "user", "content": question}]}

    async for raw_event in graph.astream(state, config={"configurable": {"env": "test"}}, stream_mode="debug", subgraphs=True):
        event = unwrap_stream_event(raw_event)
        node_name = event_node_name(event)
        if is_task_start_event(event) and node_name is not None:
            trajectory.append(node_name)
        if node_name == "tools":
            tool_names = tool_call_names(event)
            batch_id = (event_identifier(event), tuple(tool_names))
            if tool_names and batch_id not in seen_tool_batches:
                trajectory.extend(tool_names)
                seen_tool_batches.add(batch_id)
        candidate_response = response_from_event(event)
        if candidate_response is not None:
            response = candidate_response

    return {"response": response or "", "trajectory": trajectory}


def trajectory_subsequence(actual: list[str], expected: list[str]) -> float:
    if not expected:
        return 1.0
    matched = 0
    next_index = 0
    for step in actual:
        if next_index >= len(expected):
            break
        if step == expected[next_index]:
            matched += 1
            next_index += 1
    return matched / len(expected)


def unwrap_stream_event(event: object) -> object:
    if isinstance(event, tuple) and len(event) == 2 and isinstance(event[1], Mapping):
        return event[1]
    return event


def is_task_start_event(event: object) -> bool:
    if not isinstance(event, Mapping):
        return False
    return event.get("type") == "task" or event.get("event") in {"on_chain_start", "on_tool_start", "node_start"}


def event_node_name(event: object) -> str | None:
    if not isinstance(event, Mapping):
        return None
    for candidate in (mapping_value(event, "payload"), mapping_value(event, "data"), event):
        if isinstance(candidate, Mapping):
            for key in ("name", "node_name", "node"):
                if isinstance(candidate.get(key), str):
                    return candidate[key]
    metadata = mapping_value(event, "metadata")
    value = mapping_value(metadata, "langgraph_node")
    return value if isinstance(value, str) else None


def event_identifier(event: object) -> str:
    if not isinstance(event, Mapping):
        return "unknown"
    for candidate in (mapping_value(event, "payload"), mapping_value(event, "data"), event):
        if isinstance(candidate, Mapping):
            for key in ("id", "task_id", "run_id", "name"):
                if isinstance(candidate.get(key), str):
                    return candidate[key]
    return "tools"


def tool_call_names(event: object) -> list[str]:
    names: list[str] = []
    for tool_call in event_tool_calls_from_input(event) + event_messages_tool_calls(event):
        name = tool_call.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def event_messages_tool_calls(event: object) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    for message in event_messages(event):
        if isinstance(message, AIMessage):
            calls.extend(call for call in message.tool_calls if isinstance(call, Mapping))
        elif isinstance(message, Mapping) and isinstance(message.get("tool_calls"), list):
            calls.extend(call for call in message["tool_calls"] if isinstance(call, Mapping))
    return calls


def event_messages(event: object) -> list[object]:
    messages: list[object] = []
    for candidate in (mapping_value(event, "payload"), mapping_value(event, "data"), event):
        if not isinstance(candidate, Mapping):
            continue
        for key in ("input", "result", "output"):
            nested = candidate.get(key)
            if isinstance(nested, Mapping) and isinstance(nested.get("messages"), list):
                messages.extend(nested["messages"])
        if isinstance(candidate.get("messages"), list):
            messages.extend(candidate["messages"])
    return messages


def event_tool_calls_from_input(event: object) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    for candidate in (mapping_value(event, "payload"), mapping_value(event, "data"), event):
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("input"), list):
            continue
        calls.extend(
            item for item in candidate["input"]
            if isinstance(item, Mapping) and item.get("type") == "tool_call" and isinstance(item.get("name"), str)
        )
    return calls
