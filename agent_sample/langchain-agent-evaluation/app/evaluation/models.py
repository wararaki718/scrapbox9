from dataclasses import dataclass

from pydantic import BaseModel, Field


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
