from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field

TEST_CONFIG = {"configurable": {"env": "test"}}

JUDGE_SYSTEM_PROMPT = """You grade a local music-store support agent.
Decide whether the actual response correctly answers the user's question using the provided reference answer or facts.
Pass responses that are factually consistent with the reference, even when wording differs or the answer includes extra correct details.
Fail responses that contradict the reference, miss the core request, or invent unsupported facts.
Return structured output only."""


@dataclass(frozen=True)
class EvaluationExample:
    name: str
    question: str
    expected_response: str | list[str]
    expected_trajectory: list[str]
    expected_route: str
    route_messages: list[object] | None = None


class ResponseJudgeResult(BaseModel):
    passed: bool = Field(description="Whether the actual response satisfies the reference answer or facts.")
    reasoning: str = Field(description="Brief explanation for the pass/fail decision.")


EVALUATION_EXAMPLES = [
    EvaluationExample(
        name="james-brown-lookup",
        question="What James Brown songs do you have?",
        expected_response=["James Brown", "Sex Machine", "Cold Sweat"],
        expected_trajectory=["intent_classifier", "question_answering_agent", "tools", "lookup_track"],
        expected_route="question_answering_agent",
    ),
    EvaluationExample(
        name="incomplete-aaron-mitchell-refund",
        question="Please refund Aaron Mitchell's purchase of Black Dog.",
        expected_response=["phone number", "refund", "purchase"],
        expected_trajectory=["intent_classifier", "refund_agent", "respond"],
        expected_route="refund_agent",
    ),
    EvaluationExample(
        name="aaron-led-zeppelin-purchase-lookup",
        question=(
            "Please help refund Aaron Mitchell's Led Zeppelin purchase. "
            "His phone number is +1 (204) 452-6452."
        ),
        expected_response=["How Many More Times", "What Is And What Should Never Be", "2009-08-06"],
        expected_trajectory=["intent_classifier", "refund_agent", "lookup"],
        expected_route="refund_agent",
    ),
    EvaluationExample(
        name="wish-you-were-here-pink-floyd-lookup",
        question="Do you have the album Wish You Were Here by Pink Floyd?",
        expected_response=["no matches", "Wish You Were Here", "Pink Floyd"],
        expected_trajectory=["intent_classifier", "question_answering_agent", "tools", "lookup_album"],
        expected_route="question_answering_agent",
    ),
    EvaluationExample(
        name="invoice-237-refund",
        question="Refund invoice 237.",
        expected_response="Previewed a refund total of $0.99. No database changes were made because env is not prod.",
        expected_trajectory=["intent_classifier", "refund_agent", "refund"],
        expected_route="refund_agent",
    ),
]


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


def route_is_correct(actual: str, expected: str) -> bool:
    return actual == expected


async def run_graph(graph: CompiledStateGraph, question: str) -> dict[str, object]:
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config=TEST_CONFIG,
    )
    return _normalize_result(result)


async def run_with_trajectory(graph: CompiledStateGraph, question: str) -> dict[str, object]:
    trajectory: list[str] = []
    response: str | None = None
    seen_tool_batches: set[tuple[str, tuple[str, ...]]] = set()
    state = {"messages": [{"role": "user", "content": question}]}

    async for raw_event in graph.astream(
        state,
        config=TEST_CONFIG,
        stream_mode="debug",
        subgraphs=True,
    ):
        event = _unwrap_stream_event(raw_event)
        node_name = _event_node_name(event)

        if _is_task_start_event(event) and node_name is not None:
            trajectory.append(node_name)

        if node_name == "tools":
            tool_names = tuple(_tool_call_names(event))
            batch_id = (_event_identifier(event), tool_names)
            if tool_names and batch_id not in seen_tool_batches:
                trajectory.extend(tool_names)
                seen_tool_batches.add(batch_id)

        candidate_response = _response_from_event(event)
        if candidate_response is not None:
            response = candidate_response

    if response is None:
        response = str((await run_graph(graph, question)).get("response", ""))

    return {
        "response": response,
        "trajectory": trajectory,
    }


async def run_intent_classifier(graph: CompiledStateGraph, messages: Sequence[object]) -> str:
    command = await graph.nodes["intent_classifier"].ainvoke(
        {"messages": list(messages)},
        config=TEST_CONFIG,
    )
    goto = getattr(command, "goto", None)
    if isinstance(goto, str):
        return goto
    if isinstance(command, Mapping):
        mapped_goto = command.get("goto")
        if isinstance(mapped_goto, str):
            return mapped_goto
    raise ValueError("intent_classifier did not return a route")


async def run_evaluation_suite(graph: CompiledStateGraph, judge: object) -> list[dict[str, object]]:
    structured_judge = judge.with_structured_output(ResponseJudgeResult)
    results: list[dict[str, object]] = []

    for example in EVALUATION_EXAMPLES:
        trajectory_result = await run_with_trajectory(graph, example.question)
        route_messages = list(example.route_messages or [{"role": "user", "content": example.question}])
        actual_route = await run_intent_classifier(graph, route_messages)
        judged = ResponseJudgeResult.model_validate(
            await _ainvoke_or_invoke(
                structured_judge,
                [
                    SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=_judge_prompt(example, str(trajectory_result.get("response", "")))),
                ],
            )
        )
        reference_facts_matched = isinstance(example.expected_response, list) and reference_facts_present(
            example.expected_response,
            str(trajectory_result.get("response", "")),
        )
        response_correct = judged.passed or reference_facts_matched
        response_reasoning = judged.reasoning
        if reference_facts_matched and not judged.passed:
            response_reasoning = (
                "All required reference facts were present in the response. "
                f"Local judge feedback: {judged.reasoning}"
            )

        results.append(
            {
                "name": example.name,
                "question": example.question,
                "response": trajectory_result.get("response", ""),
                "expected_response": example.expected_response,
                "response_correct": response_correct,
                "response_reasoning": response_reasoning,
                "trajectory": trajectory_result.get("trajectory", []),
                "expected_trajectory": example.expected_trajectory,
                "trajectory_score": trajectory_subsequence(
                    list(trajectory_result.get("trajectory", [])),
                    example.expected_trajectory,
                ),
                "route": actual_route,
                "expected_route": example.expected_route,
                "route_correct": route_is_correct(actual_route, example.expected_route),
            }
        )

    return results


async def _ainvoke_or_invoke(runnable: object, payload: object) -> object:
    ainvoke = getattr(runnable, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(payload)

    invoke = getattr(runnable, "invoke", None)
    if callable(invoke):
        result = invoke(payload)
        if inspect.isawaitable(result):
            return await result
        return result

    raise TypeError("judge runnable must provide ainvoke or invoke")


def _judge_prompt(example: EvaluationExample, actual_response: str) -> str:
    expected_response = example.expected_response
    if isinstance(expected_response, str):
        reference = expected_response
    else:
        reference = "\n".join(f"- {fact}" for fact in expected_response)

    return (
        f"Question:\n{example.question}\n\n"
        f"Reference answer or facts:\n{reference}\n\n"
        f"Actual response:\n{actual_response}"
    )


def reference_facts_present(expected_facts: Sequence[str], actual_response: str) -> bool:
    normalized_response = actual_response.casefold()
    return bool(expected_facts) and all(
        expected_fact.casefold() in normalized_response for expected_fact in expected_facts
    )


def _normalize_result(result: object) -> dict[str, object]:
    normalized = dict(result) if isinstance(result, Mapping) else {"result": result}
    response = _normalized_text(normalized.get("followup"))
    if response is None:
        response = _message_content_to_text(_last_message(normalized.get("messages")))
    normalized["response"] = response or ""
    return normalized


def _unwrap_stream_event(event: object) -> object:
    if isinstance(event, tuple) and len(event) == 2 and isinstance(event[1], Mapping):
        return event[1]
    return event


def _is_task_start_event(event: object) -> bool:
    if not isinstance(event, Mapping):
        return False
    event_type = event.get("type")
    if event_type == "task":
        return True
    return event.get("event") in {"on_chain_start", "on_tool_start", "node_start"}


def _event_node_name(event: object) -> str | None:
    if not isinstance(event, Mapping):
        return None

    for candidate in (_mapping_value(event, "payload"), _mapping_value(event, "data"), event):
        if not isinstance(candidate, Mapping):
            continue
        for key in ("name", "node_name", "node"):
            value = candidate.get(key)
            if isinstance(value, str):
                return value
    metadata = _mapping_value(event, "metadata")
    if isinstance(metadata, Mapping):
        langgraph_node = metadata.get("langgraph_node")
        if isinstance(langgraph_node, str):
            return langgraph_node
    return None


def _event_identifier(event: object) -> str:
    if not isinstance(event, Mapping):
        return "unknown"

    for candidate in (_mapping_value(event, "payload"), _mapping_value(event, "data"), event):
        if not isinstance(candidate, Mapping):
            continue
        for key in ("id", "task_id", "run_id", "name"):
            value = candidate.get(key)
            if isinstance(value, str):
                return value
    return "tools"


def _tool_call_names(event: object) -> list[str]:
    names: list[str] = []
    for tool_call in _event_tool_calls_from_input(event):
        name = tool_call.get("name")
        if isinstance(name, str):
            names.append(name)
    for message in _event_messages(event):
        tool_calls = _message_tool_calls(message)
        for tool_call in tool_calls:
            name = tool_call.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def _event_messages(event: object) -> list[object]:
    messages: list[object] = []
    for candidate in (_mapping_value(event, "payload"), _mapping_value(event, "data"), event):
        if not isinstance(candidate, Mapping):
            continue
        for key in ("input", "result", "output"):
            nested = candidate.get(key)
            if isinstance(nested, Mapping):
                nested_messages = nested.get("messages")
                if isinstance(nested_messages, list):
                    messages.extend(nested_messages)
        direct_messages = candidate.get("messages")
        if isinstance(direct_messages, list):
            messages.extend(direct_messages)
    return messages


def _event_tool_calls_from_input(event: object) -> list[Mapping[str, Any]]:
    tool_calls: list[Mapping[str, Any]] = []
    for candidate in (_mapping_value(event, "payload"), _mapping_value(event, "data"), event):
        if not isinstance(candidate, Mapping):
            continue
        raw_input = candidate.get("input")
        if not isinstance(raw_input, list):
            continue
        for item in raw_input:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") != "tool_call":
                continue
            name = item.get("name")
            if isinstance(name, str):
                tool_calls.append(item)
    return tool_calls


def _response_from_event(event: object) -> str | None:
    for candidate in (_event_result(event), event):
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            followup = _normalized_text(candidate.get("followup"))
            if followup is not None:
                return followup
            message_text = _message_content_to_text(_last_message(candidate.get("messages")))
            if message_text is not None:
                return message_text
    return None


def _event_result(event: object) -> object:
    if not isinstance(event, Mapping):
        return None
    for container in (_mapping_value(event, "payload"), _mapping_value(event, "data"), event):
        if isinstance(container, Mapping):
            for key in ("result", "output"):
                if key in container:
                    return container[key]
    return None


def _mapping_value(mapping: object, key: str) -> object:
    if isinstance(mapping, Mapping):
        return mapping.get(key)
    return None


def _message_tool_calls(message: object) -> list[dict[str, Any]]:
    if isinstance(message, AIMessage):
        return [tool_call for tool_call in message.tool_calls if isinstance(tool_call, dict)]
    if isinstance(message, Mapping):
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            return [tool_call for tool_call in tool_calls if isinstance(tool_call, Mapping)]
    tool_calls = getattr(message, "tool_calls", None)
    if isinstance(tool_calls, list):
        return [tool_call for tool_call in tool_calls if isinstance(tool_call, Mapping)]
    return []


def _last_message(messages: object) -> AnyMessage | Mapping[str, object] | None:
    if not isinstance(messages, list) or not messages:
        return None
    last_message = messages[-1]
    if hasattr(last_message, "content") or isinstance(last_message, Mapping):
        return last_message
    return None


def _message_content_to_text(message: AnyMessage | Mapping[str, object] | None) -> str | None:
    if message is None:
        return None
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    if isinstance(content, str):
        return _normalized_text(content)
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    for block in content:
        text = _content_block_to_text(block)
        if text is not None:
            text_parts.append(text)
    if not text_parts:
        return None
    return "\n".join(text_parts)


def _content_block_to_text(block: object) -> str | None:
    if isinstance(block, str):
        return _normalized_text(block)
    if isinstance(block, Mapping):
        text = block.get("text")
        return _normalized_text(text) if isinstance(text, str) else None
    text = getattr(block, "text", None)
    return _normalized_text(text) if isinstance(text, str) else None


def _normalized_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None
