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
    ListMyOrdersArguments,
    OrderRepository,
    RequestReturnArguments,
    cancel_order,
    cancel_order_tool_definition,
    get_order_status,
    get_order_status_tool_definition,
    get_my_orders,
    get_my_orders_tool_definition,
    request_return,
    request_return_tool_definition,
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

    def handle_get_my_orders(
        arguments: BaseModel,
        context: ToolExecutionContext,
    ) -> BaseModel:
        return get_my_orders(
            ListMyOrdersArguments.model_validate(arguments),
            current_user_id=context.current_user_id,
            repository=order_repository,
        )

    def handle_request_return(arguments: BaseModel, context: ToolExecutionContext) -> BaseModel:
        return request_return(RequestReturnArguments.model_validate(arguments), context.current_user_id, context.idempotency_key or "", order_repository)

    return ToolCatalog(
        tools={
            "get_my_orders": RegisteredTool(
                definition=get_my_orders_tool_definition(),
                arguments_model=ListMyOrdersArguments,
                handler=handle_get_my_orders,
                agent_access=AgentToolAccess.READ,
            ),
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
            "request_return": RegisteredTool(definition=request_return_tool_definition(), arguments_model=RequestReturnArguments, handler=handle_request_return, effect=ToolEffect.WRITE, agent_access=AgentToolAccess.PROPOSAL),
        }
    )


DEFAULT_TOOL_CATALOG = create_tool_catalog(DEMO_ORDER_REPOSITORY)
