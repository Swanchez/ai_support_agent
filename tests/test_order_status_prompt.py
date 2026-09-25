from ai_support_agent.tools.order_status_prompt import (
    ORDER_STATUS_SYSTEM_PROMPT,
    build_order_status_tool_call,
)


def test_order_status_prompt_describes_when_the_tool_may_be_called() -> None:
    prompt = ORDER_STATUS_SYSTEM_PROMPT

    assert "только если" in prompt
    assert "ORD-<цифры>" in prompt
    assert "не подставляй" in prompt.lower()
    assert "order_not_found" in prompt


def test_build_order_status_tool_call_contains_only_rules_and_user_question() -> None:
    call = build_order_status_tool_call("Где заказ ORD-1001?")

    assert [message["role"] for message in call.messages] == ["system", "user"]
    assert call.messages[0]["content"] == ORDER_STATUS_SYSTEM_PROMPT
    assert call.messages[1]["content"] == "Где заказ ORD-1001?"
