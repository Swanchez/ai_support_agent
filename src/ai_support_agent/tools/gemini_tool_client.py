"""Manual stateless Gemini tool-calling loop with application-owned execution."""

from dataclasses import dataclass
from typing import Any

from google.genai import errors as gemini_errors
from google.genai._gaos.lib.compat_errors import GeminiNextGenAPIClientError

from ai_support_agent.config import GeminiConfig
from ai_support_agent.exceptions import LlmRequestError
from ai_support_agent.llm_client import (
    LlmCall,
    LlmResult,
    gemini_generation_config,
    gemini_response_format,
)
from ai_support_agent.tools.executor import ExecutorErrorCode, ToolExecutor
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.gemini_adapter import (
    ToolCall,
    gemini_function_result_input,
    parse_gemini_tool_turn,
)


@dataclass(frozen=True)
class ToolCallingResult:
    """One LLM result plus trusted provenance of tools actually executed by application code."""

    llm_result: LlmResult
    executed_tool_names: tuple[str, ...]
    pending_tool_calls: tuple[ToolCall, ...] = ()

    def trusted_source_ids(self) -> list[str]:
        """Return unique tool names in execution order, never trusting model-provided sources."""

        return list(dict.fromkeys(self.executed_tool_names))


@dataclass
class GeminiToolCallingClient:
    """Runs at most one application-controlled tool round with stateless history."""

    config: GeminiConfig
    sdk_client: Any | None = None

    def __post_init__(self) -> None:
        if self.sdk_client is None:
            from google import genai

            self.sdk_client = genai.Client(api_key=self.config.api_key)

    def complete_with_tools(
        self,
        call: LlmCall,
        executor: ToolExecutor,
        context: ToolExecutionContext,
    ) -> ToolCallingResult:
        """Ask Gemini for tools, execute requested calls, then request final JSON text."""

        system_instruction, user_text = _split_call(call)
        history: list[dict[str, object]] = [_user_input_step(user_text)]
        try:
            first_response = self.sdk_client.interactions.create(
                model=self.config.model,
                system_instruction=system_instruction,
                input=history,
                generation_config=gemini_generation_config(call),
                tools=executor.definitions(),
                response_format=gemini_response_format(),
                store=False,
            )
            first_turn = parse_gemini_tool_turn(first_response)
            if not first_turn.calls:
                return ToolCallingResult(
                    _result_from_responses(self.config.model, first_response),
                    (),
                )

            history.extend(_serialise_steps(first_response))
            executed_tool_names: list[str] = []
            pending_tool_calls: list[ToolCall] = []
            tool_results: list[dict[str, object]] = []
            for tool_call in first_turn.calls:
                needs_confirmation = executor.requires_confirmation(tool_call.name)
                result = executor.execute(tool_call.name, tool_call.arguments, context)
                if (
                    needs_confirmation
                    and result.get("code") == ExecutorErrorCode.CONFIRMATION_REQUIRED
                ):
                    pending_tool_calls.append(tool_call)
                elif (
                    executor.is_registered(tool_call.name)
                    and result.get("code") not in ExecutorErrorCode._value2member_map_
                ):
                    executed_tool_names.append(tool_call.name)
                tool_results.append(gemini_function_result_input(tool_call, result))
            if pending_tool_calls:
                return ToolCallingResult(
                    _result_from_responses(self.config.model, first_response),
                    tuple(executed_tool_names),
                    tuple(pending_tool_calls),
                )
            history.extend(tool_results)
            final_response = self.sdk_client.interactions.create(
                model=self.config.model,
                system_instruction=system_instruction,
                input=history,
                generation_config=gemini_generation_config(call),
                response_format=gemini_response_format(),
                store=False,
            )
        except (gemini_errors.APIError, GeminiNextGenAPIClientError) as error:
            status_code = getattr(error, "code", getattr(error, "status_code", None))
            detail = f"HTTP {status_code}" if isinstance(status_code, int) else "unknown status"
            raise LlmRequestError(f"Gemini tool request failed ({detail}).") from error

        return ToolCallingResult(
            _result_from_responses(self.config.model, first_response, final_response),
            tuple(executed_tool_names),
            tuple(pending_tool_calls),
        )


def _split_call(call: LlmCall) -> tuple[str, str]:
    """Convert the existing provider-neutral call into Gemini's system and user fields."""

    system_instruction = "\n\n".join(
        message["content"] for message in call.messages if message["role"] == "system"
    )
    user_text = "\n\n".join(
        message["content"] for message in call.messages if message["role"] != "system"
    )
    return system_instruction, user_text


def _user_input_step(text: str) -> dict[str, object]:
    """Build the first item of the stateless interaction history."""

    return {
        "type": "user_input",
        "content": [{"type": "text", "text": text}],
    }


def _serialise_steps(response: object) -> list[dict[str, object]]:
    """Keep Gemini's exact steps in locally owned stateless history."""

    serialised_steps: list[dict[str, object]] = []
    for step in getattr(response, "steps", ()):
        model_dump = getattr(step, "model_dump", None)
        if not callable(model_dump):
            raise LlmRequestError("Gemini tool step cannot be serialized for stateless history.")
        serialised = model_dump()
        if not isinstance(serialised, dict):
            raise LlmRequestError("Gemini tool step serialization must be an object.")
        serialised_steps.append(serialised)
    return serialised_steps


def _result_from_responses(model: str, *responses: object) -> LlmResult:
    """Combine usage of both stateless turns into one provider-neutral result."""

    text = getattr(responses[-1], "output_text", "") or ""
    if not isinstance(text, str):
        raise LlmRequestError("Gemini response text must be a string.")

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    for response in responses:
        usage = getattr(response, "usage", None)
        if usage is None:
            raise LlmRequestError("Gemini response did not include usage metadata.")
        input_tokens += getattr(usage, "total_input_tokens", 0) or 0
        output_tokens += getattr(usage, "total_output_tokens", 0) or 0
        total_tokens += getattr(usage, "total_tokens", 0) or 0

    return LlmResult(text, model, input_tokens, output_tokens, total_tokens)
