from pydantic import BaseModel, Field


class Answer(BaseModel):
    gatsby_line_count: int | None = Field(
        description=(
            "The verified number of lines in the complete document that contain "
            "the substring 'Gatsby'. Return null when it cannot be verified."
        ),
        ge=0,
    )
    first_daisy_line_number: int | None = Field(
        description=(
            "The verified 1-based line number of the first document line that "
            "contains 'Daisy'. Return null when it cannot be verified."
        ),
        ge=1,
    )
    synopsis: str | None = Field(
        description=(
            "A neutral synopsis of The Great Gatsby in exactly two sentences. "
            "Return null only when a synopsis cannot be provided."
        ),
    )
    how_you_computed_counts: str = Field(
        description=(
            "Explain the tool calls and results used to verify the line count "
            "and line number. Explain why a numeric field is null when "
            "verification was not possible."
        ),
    )
    errors: list[str] = Field(
        default_factory=list,
        description=(
            "Tool or execution error messages encountered while answering. "
            "Use an empty list when no errors occurred."
        ),
    )
