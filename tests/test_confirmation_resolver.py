from types import SimpleNamespace

from ai_support_agent.config import GeminiConfig
from ai_support_agent.tools.confirmation import PendingToolAction
from ai_support_agent.tools.confirmation_resolver import (
    ConfirmationDecision,
    ConfirmationResolution,
    GeminiConfirmationResolver,
)


class FakeInteractionsApi:
    def __init__(self) -> None:
        self.received_call: dict[str, object] | None = None

    def create(self, **kwargs: object) -> object:
        self.received_call = kwargs
        return SimpleNamespace(
            output_text='{"decision":"confirm"}',
            usage=SimpleNamespace(total_input_tokens=20, total_output_tokens=3, total_tokens=23),
        )


def test_confirmation_resolver_uses_strict_decision_schema() -> None:
    interactions = FakeInteractionsApi()
    resolver = GeminiConfirmationResolver(
        GeminiConfig(api_key="test-key", model="test-model"),
        sdk_client=SimpleNamespace(interactions=interactions),
    )
    pending = PendingToolAction(
        confirmation_id="confirm-1",
        user_id="user-1",
        tool_name="cancel_order",
        arguments={"order_id": "ORD-1003"},
    )

    result = resolver.resolve("Да, отменяй", pending)

    assert result.resolution.decision is ConfirmationDecision.CONFIRM
    assert result.llm_result.total_tokens == 23
    assert interactions.received_call["store"] is False
    assert interactions.received_call["response_format"]["schema"] == (
        ConfirmationResolution.model_json_schema()
    )
