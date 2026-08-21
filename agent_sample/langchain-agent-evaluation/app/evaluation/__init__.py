from .examples import EVALUATION_EXAMPLES
from .models import EvaluationExample, ResponseJudgeResult
from .response import normalize_result
from .runner import TEST_CONFIG, run_graph, run_intent_classifier
from .suite import JUDGE_SYSTEM_PROMPT, reference_facts_present, route_is_correct
from .trajectory import run_with_trajectory, trajectory_subsequence
from . import suite as _suite


async def run_evaluation_suite(graph: object, judge: object) -> list[dict[str, object]]:
    return await _suite.run_evaluation_suite(
        graph,
        judge,
        EVALUATION_EXAMPLES,
        run_with_trajectory,
        run_intent_classifier,
    )


__all__ = [
    "EVALUATION_EXAMPLES",
    "EvaluationExample",
    "JUDGE_SYSTEM_PROMPT",
    "ResponseJudgeResult",
    "TEST_CONFIG",
    "normalize_result",
    "reference_facts_present",
    "route_is_correct",
    "run_evaluation_suite",
    "run_graph",
    "run_intent_classifier",
    "run_with_trajectory",
    "trajectory_subsequence",
]
