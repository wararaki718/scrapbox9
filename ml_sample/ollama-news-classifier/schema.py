from typing import Literal

from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    category: Literal["sci.space", "rec.autos", "comp.graphics", "talk.politics.misc"]
    reason: str
