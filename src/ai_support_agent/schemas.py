from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    CLARIFICATION_NEEDED = "clarification_needed"
    CONFIRMATION_REQUIRED = "confirmation_required"


class SupportResponse(BaseModel):
    """Контракт между LLM и нашим приложением."""

    model_config = ConfigDict(extra="forbid")

    status: AnswerStatus
    answer: str = Field(min_length=1)
    alternative: str | None
    recommendations: list[str]
    sources: list[str]


def support_response_json_schema() -> dict[str, Any]:
    """Return the JSON Schema that describes a valid support response."""
    return SupportResponse.model_json_schema()


def support_response_openai_text_format() -> dict[str, Any]:
    """Return the Responses API setting for strict SupportResponse JSON."""

    return {
        "format": {
            "type": "json_schema",
            "name": "support_response",
            "strict": True,
            "schema": support_response_json_schema(),
        }
    }
