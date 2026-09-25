from dataclasses import dataclass, replace

from pydantic import BaseModel

from ai_support_agent.tools.executor import (
    DEFAULT_TOOL_EXECUTOR,
    ExecutorErrorCode,
    RegisteredTool,
    ToolEffect,
    ToolExecutor,
)
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT, ToolExecutionContext
from ai_support_agent.tools.order_status import GetOrderStatusArguments, OrderStatusSuccess


def test_executor_runs_an_explicitly_registered_tool_after_validation() -> None:
    result = DEFAULT_TOOL_EXECUTOR.execute(
        "get_order_status",
        {"order_id": "ORD-1001"},
        DEMO_TOOL_CONTEXT,
    )

    assert result["ok"] is True
    assert result["order"]["status"] == "shipped"


def test_executor_rejects_a_tool_outside_its_whitelist() -> None:
    result = DEFAULT_TOOL_EXECUTOR.execute("delete_all_orders", {}, DEMO_TOOL_CONTEXT)

    assert result == {
        "ok": False,
        "code": ExecutorErrorCode.TOOL_NOT_ALLOWED,
        "message": "Requested tool is not available.",
        "retryable": False,
    }


def test_executor_rejects_invalid_arguments_without_calling_the_handler() -> None:
    called = False

    def handler(arguments: BaseModel, context: ToolExecutionContext) -> BaseModel:
        nonlocal called
        _ = arguments, context
        called = True
        return OrderStatusSuccess.model_validate(
            {
                "order": {
                    "order_id": "ORD-1001",
                    "status": "shipped",
                    "updated_at": "2026-09-20",
                }
            }
        )

    executor = ToolExecutor(
        {"get_order_status": RegisteredTool({}, GetOrderStatusArguments, handler)}
    )

    result = executor.execute("get_order_status", {"order_id": "invalid"}, DEMO_TOOL_CONTEXT)

    assert result["code"] == ExecutorErrorCode.INVALID_TOOL_ARGUMENTS
    assert called is False


def test_executor_does_not_expose_another_users_order() -> None:
    result = DEFAULT_TOOL_EXECUTOR.execute(
        "get_order_status",
        {"order_id": "ORD-1002"},
        DEMO_TOOL_CONTEXT,
    )

    assert result["code"] == "order_not_found"


def test_executor_exposes_only_registered_public_definitions() -> None:
    definitions = DEFAULT_TOOL_EXECUTOR.definitions()

    assert [definition["name"] for definition in definitions] == [
        "get_order_status",
        "cancel_order",
    ]


def test_default_executor_blocks_cancel_order_without_confirmation() -> None:
    result = DEFAULT_TOOL_EXECUTOR.execute(
        "cancel_order",
        {"order_id": "ORD-1003"},
        DEMO_TOOL_CONTEXT,
    )

    assert result["code"] == ExecutorErrorCode.CONFIRMATION_REQUIRED


def test_executor_rejects_invalid_write_arguments_before_confirmation() -> None:
    result = DEFAULT_TOOL_EXECUTOR.execute(
        "cancel_order",
        {"order_id": "invalid"},
        DEMO_TOOL_CONTEXT,
    )

    assert result["code"] == ExecutorErrorCode.INVALID_TOOL_ARGUMENTS


def test_executor_blocks_write_tool_until_application_confirms_it() -> None:
    called = False

    def handler(arguments: BaseModel, context: ToolExecutionContext) -> BaseModel:
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

    executor = ToolExecutor(
        {
            "cancel_order": RegisteredTool(
                definition={},
                arguments_model=GetOrderStatusArguments,
                handler=handler,
                effect=ToolEffect.WRITE,
            )
        }
    )

    blocked = executor.execute(
        "cancel_order",
        {"order_id": "ORD-1001"},
        DEMO_TOOL_CONTEXT,
    )

    assert blocked["code"] == ExecutorErrorCode.CONFIRMATION_REQUIRED
    assert called is False

    completed = executor.execute(
        "cancel_order",
        {"order_id": "ORD-1001"},
        replace(DEMO_TOOL_CONTEXT, idempotency_key="confirmation-1"),
        confirmation_granted=True,
    )

    assert completed["ok"] is True
    assert called is True
