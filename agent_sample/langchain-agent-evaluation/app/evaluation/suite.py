import inspect
from collections.abc import Mapping, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from .models import EvaluationExample, ResponseJudgeResult
from .trajectory import trajectory_subsequence

JUDGE_SYSTEM_PROMPT = """You grade a local music-store support agent.
Decide whether the actual response correctly answers the user's question using the provided reference answer or facts.
Pass responses that are factually consistent with the reference, even when wording differs or the answer includes extra correct details.
Fail responses that contradict the reference, miss the core request, or invent unsupported facts.
Return structured output only."""


async def run_evaluation_suite(
    graph: object,
    judge: object,
    examples: Sequence[EvaluationExample],
    trajectory_runner,
    route_runner,
) -> list[dict[str, object]]:
    structured_judge = judge.with_structured_output(ResponseJudgeResult)
    results: list[dict[str, object]] = []
    for example in examples:
        trajectory_result = await trajectory_runner(graph, example.question)
        route_messages = list(example.route_messages or [{"role": "user", "content": example.question}])
        actual_route = await route_runner(graph, route_messages)
        judged = ResponseJudgeResult.model_validate(
            await ainvoke_or_invoke(
                structured_judge,
                [
                    SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=judge_prompt(example, str(trajectory_result.get("response", "")))),
                ],
            )
        )
        response = str(trajectory_result.get("response", ""))
        facts_matched = isinstance(example.expected_response, list) and reference_facts_present(
            example.expected_response, response
        )
        reasoning = judged.reasoning
        if facts_matched and not judged.passed:
            reasoning = f"All required reference facts were present in the response. Local judge feedback: {reasoning}"
        results.append(
            {
                "name": example.name,
                "question": example.question,
                "response": trajectory_result.get("response", ""),
                "expected_response": example.expected_response,
                "response_correct": judged.passed or facts_matched,
                "response_reasoning": reasoning,
                "trajectory": trajectory_result.get("trajectory", []),
                "expected_trajectory": example.expected_trajectory,
                "trajectory_score": trajectory_subsequence(
                    list(trajectory_result.get("trajectory", [])), example.expected_trajectory
                ),
                "route": actual_route,
                "expected_route": example.expected_route,
                "route_correct": route_is_correct(actual_route, example.expected_route),
            }
        )
    return results


async def ainvoke_or_invoke(runnable: object, payload: object) -> object:
    ainvoke = getattr(runnable, "ainvoke", None)
    if callable(ainvoke):
        return await ainvoke(payload)
    invoke = getattr(runnable, "invoke", None)
    if callable(invoke):
        result = invoke(payload)
        return await result if inspect.isawaitable(result) else result
    raise TypeError("judge runnable must provide ainvoke or invoke")


def judge_prompt(example: EvaluationExample, actual_response: str) -> str:
    expected = example.expected_response
    reference = expected if isinstance(expected, str) else "\n".join(f"- {fact}" for fact in expected)
    return f"Question:\n{example.question}\n\nReference answer or facts:\n{reference}\n\nActual response:\n{actual_response}"


def reference_facts_present(expected_facts: Sequence[str], actual_response: str) -> bool:
    normalized_response = actual_response.casefold()
    return bool(expected_facts) and all(fact.casefold() in normalized_response for fact in expected_facts)


def route_is_correct(actual: str, expected: str) -> bool:
    return actual == expected
