import pytest
from pydantic import ValidationError

from ai_support_agent.tools.order_status import (
    CancelOrderArguments,
    DEMO_ORDER_REPOSITORY,
    GetOrderStatusArguments,
    OrderRepositoryUnavailable,
    OrderStatus,
    OrderStatusFailure,
    OrderStatusSuccess,
    OrderStatusData,
    StoredOrder,
    ToolErrorCode,
    InMemoryOrderRepository,
    cancel_order,
    cancel_order_tool_definition,
    get_order_status,
    get_order_status_tool_definition,
)


def test_get_order_status_returns_raw_structured_data_for_a_known_order() -> None:
    result = get_order_status(
        GetOrderStatusArguments(order_id="ORD-1001"),
        current_user_id="demo-user-1",
    )

    assert isinstance(result, OrderStatusSuccess)
    assert result.order.status is OrderStatus.SHIPPED
    assert result.order.estimated_delivery == "2026-09-24"


def test_get_order_status_returns_a_non_retryable_not_found_result() -> None:
    result = get_order_status(
        GetOrderStatusArguments(order_id="ORD-9999"),
        current_user_id="demo-user-1",
    )

    assert isinstance(result, OrderStatusFailure)
    assert result.code is ToolErrorCode.ORDER_NOT_FOUND
    assert result.retryable is False


def test_get_order_status_returns_a_retryable_service_error() -> None:
    class UnavailableRepository:
        def find_visible_to(self, order_id: str, user_id: str) -> None:
            _ = order_id, user_id
            raise OrderRepositoryUnavailable

    result = get_order_status(
        GetOrderStatusArguments(order_id="ORD-1001"),
        current_user_id="demo-user-1",
        repository=UnavailableRepository(),
    )

    assert isinstance(result, OrderStatusFailure)
    assert result.code is ToolErrorCode.ORDER_SERVICE_UNAVAILABLE
    assert result.retryable is True


def test_get_order_status_hides_another_users_order_as_not_found() -> None:
    result = get_order_status(
        GetOrderStatusArguments(order_id="ORD-1002"),
        current_user_id="demo-user-1",
    )

    assert isinstance(result, OrderStatusFailure)
    assert result.code is ToolErrorCode.ORDER_NOT_FOUND


@pytest.mark.parametrize("order_id", ["1001", "ORD-ABC", "ORD-1001-extra"])
def test_order_status_arguments_reject_invalid_order_numbers(order_id: str) -> None:
    with pytest.raises(ValidationError):
        GetOrderStatusArguments(order_id=order_id)


def test_order_status_tool_definition_uses_the_same_argument_contract() -> None:
    definition = get_order_status_tool_definition()

    assert definition["name"] == "get_order_status"
    assert definition["parameters"] == GetOrderStatusArguments.model_json_schema()


def test_cancel_order_changes_only_an_owned_eligible_order() -> None:
    repository = InMemoryOrderRepository(
        {
            "ORD-1003": StoredOrder(
                owner_user_id="demo-user-1",
                data=OrderStatusData(
                    order_id="ORD-1003",
                    status=OrderStatus.PACKED,
                    updated_at="2026-09-22",
                ),
            )
        }
    )

    result = cancel_order(
        CancelOrderArguments(order_id="ORD-1003"),
        current_user_id="demo-user-1",
        idempotency_key="confirmation-1",
        repository=repository,
    )

    assert isinstance(result, OrderStatusSuccess)
    assert result.order.status is OrderStatus.CANCELLED
    assert repository.find_visible_to("ORD-1003", "demo-user-1").status is OrderStatus.CANCELLED


def test_cancel_order_rejects_ineligible_or_unowned_orders() -> None:
    cannot_cancel = cancel_order(
        CancelOrderArguments(order_id="ORD-1001"),
        current_user_id="demo-user-1",
        idempotency_key="confirmation-1",
    )
    not_owned = cancel_order(
        CancelOrderArguments(order_id="ORD-1002"),
        current_user_id="demo-user-1",
        idempotency_key="confirmation-2",
    )

    assert isinstance(cannot_cancel, OrderStatusFailure)
    assert cannot_cancel.code is ToolErrorCode.ORDER_CANNOT_BE_CANCELLED
    assert isinstance(not_owned, OrderStatusFailure)
    assert not_owned.code is ToolErrorCode.ORDER_NOT_FOUND


def test_cancel_order_returns_the_original_result_when_idempotency_key_repeats() -> None:
    repository = InMemoryOrderRepository(
        {
            "ORD-1003": StoredOrder(
                owner_user_id="demo-user-1",
                data=OrderStatusData(
                    order_id="ORD-1003",
                    status=OrderStatus.PACKED,
                    updated_at="2026-09-22",
                ),
            )
        }
    )

    first = cancel_order(
        CancelOrderArguments(order_id="ORD-1003"),
        current_user_id="demo-user-1",
        idempotency_key="confirmation-1",
        repository=repository,
    )
    repeated = cancel_order(
        CancelOrderArguments(order_id="ORD-1003"),
        current_user_id="demo-user-1",
        idempotency_key="confirmation-1",
        repository=repository,
    )

    assert isinstance(first, OrderStatusSuccess)
    assert repeated == first


def test_cancel_order_rejects_reusing_a_key_for_another_order() -> None:
    repository = InMemoryOrderRepository(
        {
            "ORD-1003": StoredOrder(
                owner_user_id="demo-user-1",
                data=OrderStatusData(
                    order_id="ORD-1003",
                    status=OrderStatus.PACKED,
                    updated_at="2026-09-22",
                ),
            ),
            "ORD-1004": StoredOrder(
                owner_user_id="demo-user-1",
                data=OrderStatusData(
                    order_id="ORD-1004",
                    status=OrderStatus.PACKED,
                    updated_at="2026-09-22",
                ),
            ),
        }
    )
    cancel_order(
        CancelOrderArguments(order_id="ORD-1003"),
        current_user_id="demo-user-1",
        idempotency_key="confirmation-1",
        repository=repository,
    )

    result = cancel_order(
        CancelOrderArguments(order_id="ORD-1004"),
        current_user_id="demo-user-1",
        idempotency_key="confirmation-1",
        repository=repository,
    )

    assert isinstance(result, OrderStatusFailure)
    assert result.code is ToolErrorCode.IDEMPOTENCY_KEY_CONFLICT


def test_cancel_order_definition_keeps_its_own_argument_contract() -> None:
    definition = cancel_order_tool_definition()

    assert definition["name"] == "cancel_order"
    assert definition["parameters"] == CancelOrderArguments.model_json_schema()
