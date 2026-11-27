from typing import Literal

from pydantic import BaseModel


EmotionLabel = Literal[
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "love",
    "optimism",
    "pessimism",
    "sadness",
    "surprise",
    "trust",
]

class EmotionResult(BaseModel):
    labels: list[EmotionLabel]
    reason: str
