from dataclasses import dataclass

import pytest

from ai_support_agent.exceptions import OrderNotFoundError, OrderServiceUnavailableError
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.web.order_service import ToolBackedOrderStatusService


@dataclass
class StubExecutor:
    result: dict[str, object]

    def execute(
        self,
        tool_name: str,
        arguments: object,
        context: object,
        **kwargs: object,
    ) -> dict[str, object]:
        _ = tool_name, arguments, context, kwargs
        return self.result


def test_tool_backed_order_service_returns_validated_visible_order_data() -> None:
    service = ToolBackedOrderStatusService(
        StubExecutor(
            {
                "ok": True,
                "order": {
                    "order_id": "ORD-1001",
                    "status": "shipped",
                    "updated_at": "2026-09-20",
                    "estimated_delivery": "2026-09-24",
                },
            }
        )  # type: ignore[arg-type]
    )

    order = service.get_status("ORD-1001", ToolExecutionContext("demo-user-1"))

    assert order.order_id == "ORD-1001"
    assert order.status.value == "shipped"


def test_tool_backed_order_service_maps_not_found_without_exposing_ownership() -> None:
    service = ToolBackedOrderStatusService(
        StubExecutor(
            {
                "ok": False,
                "code": "order_not_found",
                "message": "Order was not found.",
                "retryable": False,
            }
        )  # type: ignore[arg-type]
    )

    with pytest.raises(OrderNotFoundError):
        service.get_status("ORD-1001", ToolExecutionContext("demo-user-2"))


def test_tool_backed_order_service_maps_temporary_backend_failure() -> None:
    service = ToolBackedOrderStatusService(
        StubExecutor(
            {
                "ok": False,
                "code": "order_service_unavailable",
                "message": "Order service is temporarily unavailable.",
                "retryable": True,
            }
        )  # type: ignore[arg-type]
    )

    with pytest.raises(OrderServiceUnavailableError):
        service.get_status("ORD-1001", ToolExecutionContext("demo-user-1"))


def test_tool_backed_order_service_executes_cancellation_with_existing_tool_contract() -> None:
    service = ToolBackedOrderStatusService(
        StubExecutor(
            {
                "ok": True,
                "order": {
                    "order_id": "ORD-1003",
                    "status": "cancelled",
                    "updated_at": "2026-09-30",
                    "estimated_delivery": None,
                },
            }
        )  # type: ignore[arg-type]
    )

    order = service.cancel(
        "ORD-1003",
        ToolExecutionContext("demo-user-1", idempotency_key="cancel-ord-1003-v1"),
    )

    assert order.status.value == "cancelled"
