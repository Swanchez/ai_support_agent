"""Translation between Gemini Interactions steps and our provider-neutral tool calls."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One function request made by an LLM; it has not been executed yet."""

    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolTurn:
    """Relevant provider-neutral information extracted from Gemini's first tool turn."""

    text: str
    calls: tuple[ToolCall, ...]


def parse_gemini_tool_turn(response: object) -> ToolTurn:
    """Extract function-call steps without exposing Gemini SDK objects to application code."""

    calls: list[ToolCall] = []
    for step in getattr(response, "steps", ()):
        if getattr(step, "type", None) != "function_call":
            continue
        call_id = getattr(step, "id", None)
        name = getattr(step, "name", None)
        arguments = getattr(step, "arguments", None)
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Gemini function call did not include an id.")
        if not isinstance(name, str) or not name:
            raise ValueError("Gemini function call did not include a name.")
        if not isinstance(arguments, Mapping):
            raise ValueError("Gemini function call arguments must be an object.")
        calls.append(ToolCall(call_id, name, dict(arguments)))

    output_text = getattr(response, "output_text", "") or ""
    if not isinstance(output_text, str):
        raise ValueError("Gemini tool response text must be a string.")
    return ToolTurn(output_text, tuple(calls))


def gemini_function_result_input(call: ToolCall, result: Mapping[str, Any]) -> dict[str, object]:
    """Build the second-turn Interactions API payload for one executed function call."""

    return {
        "type": "function_result",
        "name": call.name,
        "call_id": call.call_id,
        "result": [
            {
                "type": "text",
                "text": json.dumps(result, ensure_ascii=False),
            }
        ],
    }
