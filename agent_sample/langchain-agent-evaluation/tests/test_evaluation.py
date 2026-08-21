from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app.evaluation as evaluation_module


class FakeGraph:
    def __init__(
        self,
        *,
        ainvoke_result: dict[str, object] | None = None,
        stream_events: list[object] | None = None,
        intent_result: object | None = None,
    ) -> None:
        self.ainvoke_result = ainvoke_result or {}
        self.stream_events = stream_events or []
        self.ainvoke_calls: list[tuple[dict[str, object], dict[str, object] | None]] = []
        self.astream_calls: list[tuple[dict[str, object], dict[str, object] | None, dict[str, object]]] = []
        self.intent_node = FakeNode(intent_result)
        self.nodes = {"intent_classifier": self.intent_node}

    async def ainvoke(self, state: dict[str, object], config: dict[str, object] | None = None):
        self.ainvoke_calls.append((state, config))
        return self.ainvoke_result

    def astream(self, state: dict[str, object], config: dict[str, object] | None = None, **kwargs):
        self.astream_calls.append((state, config, kwargs))

        async def iterator():
            for event in self.stream_events:
                yield event

        return iterator()


class FakeNode:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[dict[str, object], dict[str, object] | None]] = []

    async def ainvoke(self, state: dict[str, object], config: dict[str, object] | None = None):
        self.calls.append((state, config))
        return self.result


class FakeJudgeRunnable:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.calls: list[object] = []

    async def ainvoke(self, payload: object):
        self.calls.append(payload)
        return self.results[len(self.calls) - 1]


class FakeJudge:
    def __init__(self, results: list[object]) -> None:
        self.results = results
        self.schemas: list[object] = []
        self.runnable = FakeJudgeRunnable(results)

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self.runnable


def test_trajectory_subsequence_scores_full_match() -> None:
    assert evaluation_module.trajectory_subsequence(
        ["intent_classifier", "question_answering_agent", "tools", "lookup_track"],
        ["question_answering_agent", "lookup_track"],
    ) == 1.0


def test_trajectory_subsequence_scores_partial_match_when_expected_step_is_missing() -> None:
    assert evaluation_module.trajectory_subsequence(
        ["refund_agent"],
        ["refund_agent", "refund"],
    ) == 0.5


def test_route_is_correct_requires_exact_match() -> None:
    assert evaluation_module.route_is_correct("refund_agent", "refund_agent") is True
    assert evaluation_module.route_is_correct("refund_agent", "question_answering_agent") is False


def test_reference_facts_present_uses_case_insensitive_containment() -> None:
    assert evaluation_module.reference_facts_present(
        ["How Many More Times", "What Is And What Should Never Be", "2009-08-06"],
        "I found How Many More Times and What Is And What Should Never Be on 2009-08-06.",
    ) is True
    assert evaluation_module.reference_facts_present(
        ["How Many More Times", "2009-08-06"],
        "I found How Many More Times.",
    ) is False


def test_run_graph_uses_test_config_and_normalizes_response() -> None:
    graph = FakeGraph(
        ainvoke_result={
            "followup": "  Previewed a refund total of $0.99.  ",
            "route": "refund_agent",
        }
    )

    result = asyncio.run(evaluation_module.run_graph(graph, "Refund invoice 237."))

    assert graph.ainvoke_calls == [
        (
            {"messages": [{"role": "user", "content": "Refund invoice 237."}]},
            {"configurable": {"env": "test"}},
        )
    ]
    assert result == {
        "followup": "  Previewed a refund total of $0.99.  ",
        "route": "refund_agent",
        "response": "Previewed a refund total of $0.99.",
    }


def test_run_with_trajectory_collects_task_names_tool_calls_and_response() -> None:
    graph = FakeGraph(
        stream_events=[
            ((), {"type": "task", "payload": {"name": "intent_classifier"}}),
            {"type": "task", "payload": {"name": "question_answering_agent"}},
            ((), {"type": "task", "payload": {"name": "tools"}}),
            (
                (),
                {
                    "type": "task_result",
                    "payload": {
                        "name": "tools",
                        "result": {
                            "messages": [
                                AIMessage(
                                    content="",
                                    tool_calls=[
                                        {
                                            "name": "lookup_artist",
                                            "args": {"name": "James Brown"},
                                            "id": "call-1",
                                            "type": "tool_call",
                                        },
                                        {
                                            "name": "lookup_track",
                                            "args": {"artist": "James Brown"},
                                            "id": "call-2",
                                            "type": "tool_call",
                                        },
                                    ],
                                )
                            ]
                        },
                    },
                },
            ),
            (
                (),
                {
                    "type": "task_result",
                    "payload": {
                        "name": "compile_followup",
                        "result": {"followup": "  Found James Brown tracks.  "},
                    },
                },
            ),
        ]
    )

    result = asyncio.run(evaluation_module.run_with_trajectory(graph, "What James Brown songs do you have?"))

    assert graph.astream_calls == [
        (
            {"messages": [{"role": "user", "content": "What James Brown songs do you have?"}]},
            {"configurable": {"env": "test"}},
            {"stream_mode": "debug", "subgraphs": True},
        )
    ]
    assert result == {
        "response": "Found James Brown tracks.",
        "trajectory": [
            "intent_classifier",
            "question_answering_agent",
            "tools",
            "lookup_artist",
            "lookup_track",
        ],
    }


def test_run_with_trajectory_extracts_tool_names_from_tools_task_input_list() -> None:
    graph = FakeGraph(
        stream_events=[
            (
                (),
                {
                    "type": "task",
                    "payload": {
                        "name": "tools",
                        "input": [
                            {
                                "name": "lookup_track",
                                "args": {},
                                "id": "call1",
                                "type": "tool_call",
                            }
                        ],
                    },
                },
            ),
            (
                (),
                {
                    "type": "task_result",
                    "payload": {
                        "name": "tools",
                        "result": {"messages": [ToolMessage(content="ok", tool_call_id="call1")]},
                    },
                },
            ),
        ]
    )

    result = asyncio.run(evaluation_module.run_with_trajectory(graph, "What James Brown songs do you have?"))

    assert result["trajectory"] == ["tools", "lookup_track"]
    assert result["trajectory"].count("lookup_track") == 1


def test_run_intent_classifier_uses_direct_node_api() -> None:
    graph = FakeGraph(intent_result=Command(update={"route": "refund_agent"}, goto="refund_agent"))
    messages = [HumanMessage(content="Refund invoice 237.")]

    route = asyncio.run(evaluation_module.run_intent_classifier(graph, messages))

    assert route == "refund_agent"
    assert graph.intent_node.calls == [
        (
            {"messages": messages},
            {"configurable": {"env": "test"}},
        )
    ]


def test_run_evaluation_suite_combines_judge_trajectory_and_routes(monkeypatch) -> None:
    examples = [
        evaluation_module.EvaluationExample(
            name="james-brown-lookup",
            question="What James Brown songs do you have?",
            expected_response=["James Brown", "Cold Sweat"],
            expected_trajectory=["question_answering_agent", "lookup_track"],
            expected_route="question_answering_agent",
        ),
        evaluation_module.EvaluationExample(
            name="invoice-refund",
            question="Refund invoice 237.",
            expected_response="Previewed a refund total of $0.99.",
            expected_trajectory=["refund_agent", "refund"],
            expected_route="refund_agent",
            route_messages=[{"role": "user", "content": "Refund invoice 237."}],
        ),
    ]
    monkeypatch.setattr(evaluation_module, "EVALUATION_EXAMPLES", examples)

    run_results = iter(
        [
            {
                "response": "James Brown tracks include Cold Sweat.",
                "trajectory": ["intent_classifier", "question_answering_agent", "tools", "lookup_track"],
            },
            {
                "response": "Previewed a refund total of $0.99.",
                "trajectory": ["intent_classifier", "refund_agent"],
            },
        ]
    )
    route_calls: list[list[object]] = []

    async def fake_run_with_trajectory(graph, question: str) -> dict[str, object]:
        del graph, question
        return next(run_results)

    async def fake_run_intent_classifier(graph, messages: list[object]) -> str:
        del graph
        route_calls.append(messages)
        return "question_answering_agent" if "James Brown" in str(messages[0]) else "refund_agent"

    monkeypatch.setattr(evaluation_module, "run_with_trajectory", fake_run_with_trajectory)
    monkeypatch.setattr(evaluation_module, "run_intent_classifier", fake_run_intent_classifier)

    judge = FakeJudge(
        [
            evaluation_module.ResponseJudgeResult(passed=True, reasoning="Mentions James Brown and Cold Sweat."),
            evaluation_module.ResponseJudgeResult(passed=True, reasoning="Matches the refund preview exactly."),
        ]
    )

    results = asyncio.run(evaluation_module.run_evaluation_suite(object(), judge))

    assert judge.schemas == [evaluation_module.ResponseJudgeResult]
    assert "Cold Sweat" in str(judge.runnable.calls[0])
    assert "Previewed a refund total of $0.99." in str(judge.runnable.calls[1])
    assert route_calls == [
        [{"role": "user", "content": "What James Brown songs do you have?"}],
        [{"role": "user", "content": "Refund invoice 237."}],
    ]
    assert results == [
        {
            "name": "james-brown-lookup",
            "question": "What James Brown songs do you have?",
            "response": "James Brown tracks include Cold Sweat.",
            "expected_response": ["James Brown", "Cold Sweat"],
            "response_correct": True,
            "response_reasoning": "Mentions James Brown and Cold Sweat.",
            "trajectory": ["intent_classifier", "question_answering_agent", "tools", "lookup_track"],
            "expected_trajectory": ["question_answering_agent", "lookup_track"],
            "trajectory_score": 1.0,
            "route": "question_answering_agent",
            "expected_route": "question_answering_agent",
            "route_correct": True,
        },
        {
            "name": "invoice-refund",
            "question": "Refund invoice 237.",
            "response": "Previewed a refund total of $0.99.",
            "expected_response": "Previewed a refund total of $0.99.",
            "response_correct": True,
            "response_reasoning": "Matches the refund preview exactly.",
            "trajectory": ["intent_classifier", "refund_agent"],
            "expected_trajectory": ["refund_agent", "refund"],
            "trajectory_score": 0.5,
            "route": "refund_agent",
            "expected_route": "refund_agent",
            "route_correct": True,
        },
    ]
