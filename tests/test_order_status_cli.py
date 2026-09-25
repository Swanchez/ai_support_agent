from ai_support_agent.order_status_cli import (
    attach_trusted_tool_sources,
    format_support_response,
)
from ai_support_agent.schemas import AnswerStatus, SupportResponse


def test_format_support_response_keeps_only_user_facing_fields() -> None:
    response = SupportResponse(
        status=AnswerStatus.ANSWERED,
        answer="Order is shipped.",
        alternative="Reference information.",
        recommendations=["Wait for delivery."],
        sources=["order-service"],
    )

    rendered = format_support_response(response)

    assert "Order is shipped." in rendered
    assert "Reference information." in rendered
    assert "Рекомендации:" in rendered
    assert "Источники:" in rendered
    assert "answered" not in rendered


def test_attach_trusted_tool_sources_replaces_model_claimed_sources() -> None:
    response = SupportResponse(
        status=AnswerStatus.ANSWERED,
        answer="Order is shipped.",
        alternative=None,
        recommendations=[],
        sources=["made-up-source"],
    )

    trusted_response = attach_trusted_tool_sources(response, ["get_order_status"])

    assert trusted_response.sources == ["get_order_status"]
