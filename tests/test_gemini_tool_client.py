from types import SimpleNamespace

from ai_support_agent.config import GeminiConfig
from ai_support_agent.llm_client import LlmCall, gemini_response_format
from pydantic import BaseModel

from ai_support_agent.tools.executor import (
    DEFAULT_TOOL_EXECUTOR,
    RegisteredTool,
    ToolEffect,
    ToolExecutor,
)
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.order_status import GetOrderStatusArguments, OrderStatusSuccess
from ai_support_agent.tools.gemini_tool_client import GeminiToolCallingClient


class FakeStep:
    def __init__(self, payload: dict[str, object]) -> None:
        self.type = payload["type"]
        self.id = payload.get("id")
        self.name = payload.get("name")
        self.arguments = payload.get("arguments")
        self.payload = payload

    def model_dump(self) -> dict[str, object]:
        return self.payload


class FakeInteractionsApi:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.received_calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.received_calls.append(kwargs)
        return self.responses.pop(0)


def response(*, output_text: str, steps: list[FakeStep], input_tokens: int) -> object:
    return SimpleNamespace(
        id="interaction-1",
        output_text=output_text,
        steps=steps,
        usage=SimpleNamespace(
            total_input_tokens=input_tokens,
            total_output_tokens=10,
            total_tokens=input_tokens + 10,
        ),
    )


def test_gemini_tool_client_runs_stateless_two_turn_cycle() -> None:
    first_response = response(
        output_text="",
        input_tokens=30,
        steps=[
            FakeStep(
                {
                    "type": "function_call",
                    "id": "call-1",
                    "name": "get_order_status",
                    "arguments": {"order_id": "ORD-1001"},
                }
            )
        ],
    )
    final_response = response(output_text='{"status": "answered"}', input_tokens=50, steps=[])
    interactions = FakeInteractionsApi([first_response, final_response])
    client = GeminiToolCallingClient(
        GeminiConfig(api_key="test-key", model="test-model"),
        sdk_client=SimpleNamespace(interactions=interactions),
    )

    result = client.complete_with_tools(
        LlmCall(messages=[{"role": "system", "content": "rules"}, {"role": "user", "content": "ORD-1001"}]),
        DEFAULT_TOOL_EXECUTOR,
        DEMO_TOOL_CONTEXT,
    )

    assert result.llm_result.text == '{"status": "answered"}'
    assert result.llm_result.input_tokens == 80
    assert result.trusted_source_ids() == ["get_order_status"]
    assert len(interactions.received_calls) == 2
    first_call, second_call = interactions.received_calls
    assert first_call["store"] is False
    assert first_call["tools"] == DEFAULT_TOOL_EXECUTOR.definitions()
    assert first_call["response_format"] == gemini_response_format()
    assert second_call["store"] is False
    assert "previous_interaction_id" not in second_call
    assert second_call["response_format"] == gemini_response_format()
    assert second_call["input"][1]["type"] == "function_call"
    assert second_call["input"][2]["type"] == "function_result"
    assert "shipped" in second_call["input"][2]["result"][0]["text"]


def test_gemini_tool_client_returns_structured_first_response_when_no_tool_is_called() -> None:
    first_response = response(
        output_text='{"status": "clarification_needed", "answer": "Укажите номер заказа."}',
        input_tokens=30,
        steps=[],
    )
    interactions = FakeInteractionsApi([first_response])
    client = GeminiToolCallingClient(
        GeminiConfig(api_key="test-key", model="test-model"),
        sdk_client=SimpleNamespace(interactions=interactions),
    )

    result = client.complete_with_tools(
        LlmCall(messages=[{"role": "user", "content": "Hello"}]),
        DEFAULT_TOOL_EXECUTOR,
        DEMO_TOOL_CONTEXT,
    )

    assert result.llm_result.text == (
        '{"status": "clarification_needed", "answer": "Укажите номер заказа."}'
    )
    assert result.trusted_source_ids() == []
    assert len(interactions.received_calls) == 1
    assert interactions.received_calls[0]["response_format"] == gemini_response_format()


def test_gemini_tool_client_marks_write_call_as_pending_without_running_handler() -> None:
    called = False

    def handler(arguments: BaseModel, context: object) -> BaseModel:
        nonlocal called
        _ = arguments, context
        called = True
        return OrderStatusSuccess.model_validate(
            {
                "order": {
                    "order_id": "ORD-1001",
                    "status": "cancelled",
                    "updated_at": "2026-09-22",
                }
            }
        )

    write_executor = ToolExecutor(
        {
            "cancel_order": RegisteredTool(
                definition={"name": "cancel_order"},
                arguments_model=GetOrderStatusArguments,
                handler=handler,
                effect=ToolEffect.WRITE,
            )
        }
    )
    first_response = response(
        output_text="",
        input_tokens=30,
        steps=[
            FakeStep(
                {
                    "type": "function_call",
                    "id": "call-1",
                    "name": "cancel_order",
                    "arguments": {"order_id": "ORD-1001"},
                }
            )
        ],
    )
    final_response = response(
        output_text='{"status":"clarification_needed","answer":"Confirm?"}',
        input_tokens=50,
        steps=[],
    )
    client = GeminiToolCallingClient(
        GeminiConfig(api_key="test-key", model="test-model"),
        sdk_client=SimpleNamespace(interactions=FakeInteractionsApi([first_response, final_response])),
    )

    result = client.complete_with_tools(
        LlmCall(messages=[{"role": "user", "content": "Cancel ORD-1001"}]),
        write_executor,
        DEMO_TOOL_CONTEXT,
    )

    assert result.trusted_source_ids() == []
    assert result.pending_tool_calls[0].name == "cancel_order"
    assert called is False
    assert len(client.sdk_client.interactions.received_calls) == 1
