"""Semantic resolution of a user's reply to one pending write action."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from google.genai import errors as gemini_errors
from google.genai._gaos.lib.compat_errors import GeminiNextGenAPIClientError
from pydantic import BaseModel, ConfigDict, ValidationError

from ai_support_agent.config import GeminiConfig
from ai_support_agent.exceptions import InvalidModelResponseError, LlmRequestError
from ai_support_agent.llm_client import LlmResult, gemini_json_response_format
from ai_support_agent.tools.confirmation import PendingToolAction


class ConfirmationDecision(StrEnum):
    """The only outcomes allowed for a free-form confirmation reply."""

    CONFIRM = "confirm"
    REJECT = "reject"
    UNCLEAR = "unclear"


class ConfirmationResolution(BaseModel):
    """Validated classification result; it never carries tool arguments."""

    model_config = ConfigDict(extra="forbid")

    decision: ConfirmationDecision


@dataclass(frozen=True)
class ConfirmationResolutionResult:
    """Decision plus provider-reported usage for the classification request."""

    resolution: ConfirmationResolution
    llm_result: LlmResult


class ConfirmationResolver(Protocol):
    """Boundary for resolving natural-language confirmation messages."""

    def resolve(
        self,
        user_reply: str,
        pending_action: PendingToolAction,
    ) -> ConfirmationResolutionResult:
        """Classify one reply against one application-owned pending action."""


@dataclass
class GeminiConfirmationResolver:
    """Gemini adapter that classifies confirmation intent without executing tools."""

    config: GeminiConfig
    sdk_client: Any | None = None

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            from google import genai

            self.sdk_client = genai.Client(api_key=self.config.api_key)

    def resolve(
        self,
        user_reply: str,
        pending_action: PendingToolAction,
    ) -> ConfirmationResolutionResult:
        """Return a strict semantic decision for the saved action only."""

        prompt = (
            "Определи, подтверждает ли пользователь сохранённое действие. "
            "Верни confirm только при однозначном согласии именно с этим действием. "
            "Верни reject при явном отказе. Верни unclear для нового действия, другого номера "
            "заказа, вопроса или неоднозначного ответа.\n\n"
            f"Сохранённое действие: {pending_action.tool_name}\n"
            f"Аргументы: {pending_action.arguments}\n"
            f"Ответ пользователя: {user_reply}"
        )
        try:
            response = self.sdk_client.interactions.create(
                model=self.config.model,
                input=prompt,
                response_format=gemini_json_response_format(
                    ConfirmationResolution.model_json_schema()
                ),
                store=False,
            )
        except (gemini_errors.APIError, GeminiNextGenAPIClientError) as error:
            raise LlmRequestError("Gemini confirmation request failed.") from error

        try:
            resolution = ConfirmationResolution.model_validate_json(response.output_text or "")
        except ValidationError as error:
            raise InvalidModelResponseError(
                "Confirmation response does not match its contract."
            ) from error

        usage = response.usage
        if usage is None:
            raise LlmRequestError("Gemini confirmation response did not include usage metadata.")
        return ConfirmationResolutionResult(
            resolution=resolution,
            llm_result=LlmResult(
                text=response.output_text or "",
                model=self.config.model,
                input_tokens=usage.total_input_tokens or 0,
                output_tokens=usage.total_output_tokens or 0,
                total_tokens=usage.total_tokens or 0,
            ),
        )
