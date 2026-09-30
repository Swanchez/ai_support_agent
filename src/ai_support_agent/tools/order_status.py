"""A safe, local order-status tool used to learn tool-calling contracts."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class OrderStatus(StrEnum):
    """The finite set of order states that our application understands."""

    PROCESSING = "processing"
    PACKED = "packed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class ToolErrorCode(StrEnum):
    """Machine-readable failures that the application can handle predictably."""

    ORDER_NOT_FOUND = "order_not_found"
    ORDER_CANNOT_BE_CANCELLED = "order_cannot_be_cancelled"
    ORDER_SERVICE_UNAVAILABLE = "order_service_unavailable"
    IDEMPOTENCY_KEY_CONFLICT = "idempotency_key_conflict"


class GetOrderStatusArguments(BaseModel):
    """The only model-provided argument accepted by this tool."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(
        min_length=6,
        max_length=32,
        pattern=r"^ORD-\d+$",
        description="Public order number in the format ORD-12345.",
    )


class CancelOrderArguments(GetOrderStatusArguments):
    """Validated arguments for a requested cancellation of one specific order."""


class OrderStatusData(BaseModel):
    """Raw business data returned by the order service, not user-facing prose."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    status: OrderStatus
    updated_at: str
    estimated_delivery: str | None = None


@dataclass(frozen=True)
class StoredOrder:
    """Private repository record; its owner must never be exposed to the model."""

    owner_user_id: str
    data: OrderStatusData


class OrderStatusSuccess(BaseModel):
    """Successful structured result of get_order_status."""

    model_config = ConfigDict(extra="forbid")

    ok: Literal[True] = True
    order: OrderStatusData


class OrderStatusFailure(BaseModel):
    """Expected operational failure returned as data rather than an exception."""

    model_config = ConfigDict(extra="forbid")

    ok: Literal[False] = False
    code: ToolErrorCode
    message: str
    retryable: bool


OrderStatusResult: TypeAlias = OrderStatusSuccess | OrderStatusFailure

CANCELLABLE_ORDER_STATUSES = frozenset({OrderStatus.PROCESSING, OrderStatus.PACKED})


class OrderRepositoryUnavailable(RuntimeError):
    """The backing order service could not be reached temporarily."""


class IdempotencyKeyConflict(RuntimeError):
    """One key was reused for a request with different arguments."""


class OrderCancellationNotAllowed(RuntimeError):
    """The order changed state after the pre-check and can no longer be cancelled."""


class VisibleOrderRepository(Protocol):
    """Read boundary that exposes an order only to its authenticated owner."""

    def find_visible_to(self, order_id: str, user_id: str) -> OrderStatusData | None:
        """Return an order only when it belongs to the authenticated user."""


class OrderRepository(VisibleOrderRepository, Protocol):
    """Full order boundary used by confirmation-gated write operations."""

    def cancel_visible_to(
        self,
        order_id: str,
        user_id: str,
        idempotency_key: str,
    ) -> OrderStatusData | None:
        """Cancel an eligible visible order and return its new state."""

    def find_cancellation_result(
        self,
        user_id: str,
        idempotency_key: str,
        order_id: str,
    ) -> OrderStatusData | None:
        """Return the earlier successful cancellation for one idempotency key."""


@dataclass
class InMemoryOrderRepository:
    """Deterministic local order data; it never makes a network request."""

    orders: dict[str, StoredOrder]
    cancellation_results: dict[tuple[str, str], OrderStatusData] = field(default_factory=dict)

    def find_visible_to(self, order_id: str, user_id: str) -> OrderStatusData | None:
        order = self.orders.get(order_id)
        if order is None or order.owner_user_id != user_id:
            return None
        return order.data

    def cancel_visible_to(
        self,
        order_id: str,
        user_id: str,
        idempotency_key: str,
    ) -> OrderStatusData | None:
        previous_result = self.cancellation_results.get((user_id, idempotency_key))
        if previous_result is not None:
            if previous_result.order_id != order_id:
                raise IdempotencyKeyConflict("Idempotency key belongs to another order.")
            return previous_result
        order = self.orders.get(order_id)
        if order is None or order.owner_user_id != user_id:
            return None
        if order.data.status not in CANCELLABLE_ORDER_STATUSES:
            raise OrderCancellationNotAllowed("Order cannot be cancelled in its current status.")
        cancelled = order.data.model_copy(update={"status": OrderStatus.CANCELLED})
        self.orders[order_id] = StoredOrder(order.owner_user_id, cancelled)
        self.cancellation_results[(user_id, idempotency_key)] = cancelled
        return cancelled

    def find_cancellation_result(
        self,
        user_id: str,
        idempotency_key: str,
        order_id: str,
    ) -> OrderStatusData | None:
        previous_result = self.cancellation_results.get((user_id, idempotency_key))
        if previous_result is not None and previous_result.order_id != order_id:
            raise IdempotencyKeyConflict("Idempotency key belongs to another order.")
        return previous_result


DEMO_ORDER_REPOSITORY = InMemoryOrderRepository(
    orders={
        "ORD-1001": StoredOrder(
            owner_user_id="demo-user-1",
            data=OrderStatusData(
                order_id="ORD-1001",
                status=OrderStatus.SHIPPED,
                updated_at="2026-09-20",
                estimated_delivery="2026-09-24",
            ),
        ),
        "ORD-1002": StoredOrder(
            owner_user_id="demo-user-2",
            data=OrderStatusData(
                order_id="ORD-1002",
                status=OrderStatus.PROCESSING,
                updated_at="2026-09-21",
            ),
        ),
        "ORD-1003": StoredOrder(
            owner_user_id="demo-user-1",
            data=OrderStatusData(
                order_id="ORD-1003",
                status=OrderStatus.PACKED,
                updated_at="2026-09-22",
            ),
        ),
    }
)


def get_order_status(
    arguments: GetOrderStatusArguments,
    current_user_id: str,
    repository: VisibleOrderRepository = DEMO_ORDER_REPOSITORY,
) -> OrderStatusResult:
    """Look up a validated order only within the authenticated user's visibility."""

    try:
        order = repository.find_visible_to(arguments.order_id, current_user_id)
    except OrderRepositoryUnavailable:
        return OrderStatusFailure(
            code=ToolErrorCode.ORDER_SERVICE_UNAVAILABLE,
            message="Order service is temporarily unavailable.",
            retryable=True,
        )

    if order is None:
        return OrderStatusFailure(
            code=ToolErrorCode.ORDER_NOT_FOUND,
            message="Order was not found.",
            retryable=False,
        )
    return OrderStatusSuccess(order=order)


def cancel_order(
    arguments: CancelOrderArguments,
    current_user_id: str,
    idempotency_key: str,
    repository: OrderRepository = DEMO_ORDER_REPOSITORY,
) -> OrderStatusResult:
    """Cancel one owned order only when its current state permits cancellation."""

    try:
        previous_result = repository.find_cancellation_result(
            current_user_id,
            idempotency_key,
            arguments.order_id,
        )
        if previous_result is not None:
            return OrderStatusSuccess(order=previous_result)
        order = repository.find_visible_to(arguments.order_id, current_user_id)
        if order is None:
            return OrderStatusFailure(
                code=ToolErrorCode.ORDER_NOT_FOUND,
                message="Order was not found.",
                retryable=False,
            )
        if order.status not in CANCELLABLE_ORDER_STATUSES:
            return OrderStatusFailure(
                code=ToolErrorCode.ORDER_CANNOT_BE_CANCELLED,
                message="Order cannot be cancelled in its current status.",
                retryable=False,
            )
        cancelled = repository.cancel_visible_to(
            arguments.order_id,
            current_user_id,
            idempotency_key,
        )
    except IdempotencyKeyConflict:
        return OrderStatusFailure(
            code=ToolErrorCode.IDEMPOTENCY_KEY_CONFLICT,
            message="Idempotency key conflicts with another request.",
            retryable=False,
        )
    except OrderCancellationNotAllowed:
        return OrderStatusFailure(
            code=ToolErrorCode.ORDER_CANNOT_BE_CANCELLED,
            message="Order cannot be cancelled in its current status.",
            retryable=False,
        )
    except OrderRepositoryUnavailable:
        return OrderStatusFailure(
            code=ToolErrorCode.ORDER_SERVICE_UNAVAILABLE,
            message="Order service is temporarily unavailable.",
            retryable=True,
        )

    if cancelled is None:
        return OrderStatusFailure(
            code=ToolErrorCode.ORDER_NOT_FOUND,
            message="Order was not found.",
            retryable=False,
        )
    return OrderStatusSuccess(order=cancelled)


def get_order_status_tool_definition() -> dict[str, object]:
    """Describe the tool for a future provider-specific tool-calling adapter."""

    return {
        "type": "function",
        "name": "get_order_status",
        "description": "Gets current status of one order by its public order number.",
        "parameters": GetOrderStatusArguments.model_json_schema(),
    }


def cancel_order_tool_definition() -> dict[str, object]:
    """Describe the confirmation-protected cancellation tool for an LLM provider."""

    return {
        "type": "function",
        "name": "cancel_order",
        "description": "Cancels one eligible order after explicit user confirmation.",
        "parameters": CancelOrderArguments.model_json_schema(),
    }
