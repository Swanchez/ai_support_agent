import json
from dataclasses import dataclass
from typing import Any, Protocol

from google.genai import errors as gemini_errors
from google.genai._gaos.lib.compat_errors import GeminiNextGenAPIClientError
from openai import APIError

from ai_support_agent.config import GeminiConfig, OpenAiConfig
from ai_support_agent.exceptions import LlmRequestError
from ai_support_agent.schemas import (
    support_response_json_schema,
    support_response_openai_text_format,
)


OPENAI_TIMEOUT_SECONDS = 20.0
OPENAI_MAX_RETRIES = 0


@dataclass(frozen=True)
class LlmCall:
    """All data required for one request to an LLM provider."""

    messages: list[dict[str, str]]
    temperature: float = 0.2
    max_output_tokens: int = 200


class LlmClient(Protocol):
    """A provider-independent boundary for invoking an LLM."""

    def complete(self, call: LlmCall) -> "LlmResult":
        """Return the model text together with usage metadata."""


@dataclass(frozen=True)
class LlmResult:
    """The raw model response and the usage reported by its provider."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


def openai_sdk_options(api_key: str) -> dict[str, str | float | int]:
    """Return explicit, conservative transport settings for the OpenAI SDK."""

    return {
        "api_key": api_key,
        "timeout": OPENAI_TIMEOUT_SECONDS,
        "max_retries": OPENAI_MAX_RETRIES,
    }


@dataclass
class OpenAiLlmClient:
    """Adapter from the OpenAI Responses API to the application's contract."""

    config: OpenAiConfig
    sdk_client: Any | None = None

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            from openai import OpenAI

            self.sdk_client = OpenAI(**openai_sdk_options(self.config.api_key))

    def complete(self, call: LlmCall) -> LlmResult:
        """Send one call to OpenAI and return provider-neutral data."""

        system_messages = [
            message["content"] for message in call.messages if message["role"] == "system"
        ]
        input_messages = [
            message for message in call.messages if message["role"] != "system"
        ]

        try:
            response = self.sdk_client.responses.create(
                model=self.config.model,
                instructions="\n\n".join(system_messages),
                input=input_messages,
                temperature=call.temperature,
                max_output_tokens=call.max_output_tokens,
                store=False,
                text=support_response_openai_text_format(),
            )
        except APIError as error:
            raise LlmRequestError("OpenAI request failed.") from error

        usage = response.usage
        return LlmResult(
            text=response.output_text,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )


def gemini_generation_config(call: LlmCall) -> dict[str, object]:
    """Translate our generation controls into Gemini settings."""

    return {
        "temperature": call.temperature,
        "max_output_tokens": call.max_output_tokens,
    }


def gemini_json_response_format(schema: dict[str, object]) -> dict[str, object]:
    """Build a Gemini structured-output format from an application-owned schema."""

    return {
        "type": "text",
        "mime_type": "application/json",
        "schema": schema,
    }


def gemini_response_format() -> dict[str, object]:
    """Translate the standard support-response contract into Gemini's format."""

    return gemini_json_response_format(support_response_json_schema())


@dataclass
class GeminiLlmClient:
    """Adapter from the Gemini Interactions API to the application's contract."""

    config: GeminiConfig
    sdk_client: Any | None = None

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            from google import genai

            self.sdk_client = genai.Client(api_key=self.config.api_key)

    def complete(self, call: LlmCall) -> LlmResult:
        """Send one call to Gemini and return provider-neutral data."""

        system_instruction = "\n\n".join(
            message["content"] for message in call.messages if message["role"] == "system"
        )
        user_input = "\n\n".join(
            message["content"] for message in call.messages if message["role"] != "system"
        )

        try:
            response = self.sdk_client.interactions.create(
                model=self.config.model,
                system_instruction=system_instruction,
                input=user_input,
                generation_config=gemini_generation_config(call),
                response_format=gemini_response_format(),
                store=False,
            )
        except (gemini_errors.APIError, GeminiNextGenAPIClientError) as error:
            status_code = getattr(error, "code", getattr(error, "status_code", None))
            detail = f"HTTP {status_code}" if isinstance(status_code, int) else "unknown status"
            raise LlmRequestError(f"Gemini request failed ({detail}).") from error

        usage = response.usage
        if usage is None:
            raise LlmRequestError("Gemini response did not include usage metadata.")

        return LlmResult(
            text=response.output_text or "",
            model=self.config.model,
            input_tokens=usage.total_input_tokens or 0,
            output_tokens=usage.total_output_tokens or 0,
            total_tokens=usage.total_tokens or 0,
        )


@dataclass(frozen=True)
class FakeLlmClient:
    """A deterministic offline client for learning and automated tests."""

    raw_response: str | None = None

    def complete(self, call: LlmCall) -> LlmResult:
        _ = call
        if self.raw_response is not None:
            text = self.raw_response
        else:
            text = json.dumps(
                {
                    "status": "insufficient_context",
                    "answer": (
                        "Точную дату поступления денег назвать не могу. "
                        "После поступления товара на склад возврат обычно "
                        "обрабатывается в течение 3–7 рабочих дней."
                    ),
                    "alternative": None,
                    "recommendations": ["Проверьте статус поступления товара на склад."],
                    "sources": ["refund-policy-v1"],
                },
                ensure_ascii=False,
            )

        return LlmResult(
            text=text,
            model="fake",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
        )
