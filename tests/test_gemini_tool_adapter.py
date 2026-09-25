from types import SimpleNamespace

import pytest

from ai_support_agent.tools.gemini_adapter import (
    ToolCall,
    gemini_function_result_input,
    parse_gemini_tool_turn,
)


def test_parse_gemini_tool_turn_extracts_only_function_calls() -> None:
    response = SimpleNamespace(
        id="interaction-1",
        output_text="",
        steps=[
            SimpleNamespace(type="thought", content="Need order status"),
            SimpleNamespace(
                type="function_call",
                id="call-1",
                name="get_order_status",
                arguments={"order_id": "ORD-1001"},
            ),
        ],
    )

    turn = parse_gemini_tool_turn(response)

    assert turn.calls == (ToolCall("call-1", "get_order_status", {"order_id": "ORD-1001"}),)


def test_parse_gemini_tool_turn_rejects_a_call_without_an_id() -> None:
    response = SimpleNamespace(
        id="interaction-1",
        output_text="",
        steps=[SimpleNamespace(type="function_call", id=None, name="tool", arguments={})],
    )

    with pytest.raises(ValueError, match="did not include an id"):
        parse_gemini_tool_turn(response)


def test_gemini_function_result_input_preserves_call_identity_and_json_result() -> None:
    payload = gemini_function_result_input(
        ToolCall("call-1", "get_order_status", {"order_id": "ORD-1001"}),
        {"ok": True, "order": {"status": "shipped"}},
    )

    assert payload["type"] == "function_result"
    assert payload["name"] == "get_order_status"
    assert payload["call_id"] == "call-1"
    assert '"status": "shipped"' in payload["result"][0]["text"]
