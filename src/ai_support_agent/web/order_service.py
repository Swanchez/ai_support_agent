"""HTTP-facing application service for safe order-status reads."""

from dataclasses import dataclass

from pydantic import ValidationError

from ai_support_agent.exceptions import (
    OrderCancellationConflictError,
    OrderNotFoundError,
    OrderServiceUnavailableError,
)
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.executor import ToolExecutor
from ai_support_agent.tools.order_status import (
    OrderStatusData,
    OrderStatusFailure,
    OrderStatusSuccess,
    ToolErrorCode,
)


@dataclass(frozen=True)
class ToolBackedOrderStatusService:
    """Reuse the application-owned tool boundary for one authenticated API request."""

    executor: ToolExecutor

    def get_status(self, order_id: str, context: ToolExecutionContext) -> OrderStatusData:
        """Return only a visible order or translate safe tool results into app errors."""

        raw_result = self.executor.execute(
            "get_order_status",
            {"order_id": order_id},
            context,
        )
        return _read_order_result(raw_result)

    def cancel(self, order_id: str, context: ToolExecutionContext) -> OrderStatusData:
        """Execute the explicit HTTP cancellation action through the trusted tool boundary."""

        raw_result = self.executor.execute(
            "cancel_order",
            {"order_id": order_id},
            context,
            confirmation_granted=True,
        )
        return _read_order_result(raw_result)


def _read_order_result(raw_result: dict[str, object]) -> OrderStatusData:
    """Map the existing safe tool result to HTTP-neutral application outcomes."""

    try:
        if raw_result.get("ok") is True:
            return OrderStatusSuccess.model_validate(raw_result).order
        failure = OrderStatusFailure.model_validate(raw_result)
    except ValidationError as error:
        raise OrderServiceUnavailableError("Order tool returned an invalid result.") from error

    if failure.code is ToolErrorCode.ORDER_NOT_FOUND:
        raise OrderNotFoundError("Order was not found.")
    if failure.code is ToolErrorCode.ORDER_SERVICE_UNAVAILABLE:
        raise OrderServiceUnavailableError("Order service is temporarily unavailable.")
    if failure.code in {
        ToolErrorCode.ORDER_CANNOT_BE_CANCELLED,
        ToolErrorCode.IDEMPOTENCY_KEY_CONFLICT,
    }:
        raise OrderCancellationConflictError("Order cancellation conflicts with current state.")
    raise OrderServiceUnavailableError("Order tool could not complete the request.")
