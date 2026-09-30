"""Application-specific registrations for the tools currently supported by the demo."""

from pydantic import BaseModel

from ai_support_agent.tools.catalog import (
    AgentToolAccess,
    RegisteredTool,
    ToolCatalog,
    ToolEffect,
)
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.order_status import (
    CancelOrderArguments,
    DEMO_ORDER_REPOSITORY,
    GetOrderStatusArguments,
    OrderRepository,
    cancel_order,
    cancel_order_tool_definition,
    get_order_status,
    get_order_status_tool_definition,
)


def create_tool_catalog(order_repository: OrderRepository) -> ToolCatalog:
    """Build the same allowed tools around one chosen order-data implementation."""

    def handle_get_order_status(
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> BaseModel:
        return get_order_status(
            GetOrderStatusArguments.model_validate(arguments),
            current_user_id=context.current_user_id,
            repository=order_repository,
        )

    def handle_cancel_order(
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> BaseModel:
        return cancel_order(
            CancelOrderArguments.model_validate(arguments),
            current_user_id=context.current_user_id,
            idempotency_key=context.idempotency_key,
            repository=order_repository,
        )

    return ToolCatalog(
        tools={
            "get_order_status": RegisteredTool(
                definition=get_order_status_tool_definition(),
                arguments_model=GetOrderStatusArguments,
                handler=handle_get_order_status,
                agent_access=AgentToolAccess.READ,
            ),
            "cancel_order": RegisteredTool(
                definition=cancel_order_tool_definition(),
                arguments_model=CancelOrderArguments,
                handler=handle_cancel_order,
                effect=ToolEffect.WRITE,
                agent_access=AgentToolAccess.PROPOSAL,
            ),
        }
    )


DEFAULT_TOOL_CATALOG = create_tool_catalog(DEMO_ORDER_REPOSITORY)
