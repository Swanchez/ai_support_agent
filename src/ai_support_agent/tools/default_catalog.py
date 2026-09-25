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
    GetOrderStatusArguments,
    cancel_order,
    cancel_order_tool_definition,
    get_order_status,
    get_order_status_tool_definition,
)


def _handle_get_order_status(
    arguments: BaseModel,
    context: ToolExecutionContext,
) -> BaseModel:
    """Adapt the executor's generic boundary to the typed order-status handler."""

    return get_order_status(
        GetOrderStatusArguments.model_validate(arguments),
        current_user_id=context.current_user_id,
    )


def _handle_cancel_order(
    arguments: BaseModel,
    context: ToolExecutionContext,
) -> BaseModel:
    """Call the write handler only after executor confirmation and idempotency gates."""

    return cancel_order(
        CancelOrderArguments.model_validate(arguments),
        current_user_id=context.current_user_id,
        idempotency_key=context.idempotency_key,
    )


DEFAULT_TOOL_CATALOG = ToolCatalog(
    tools={
        "get_order_status": RegisteredTool(
            definition=get_order_status_tool_definition(),
            arguments_model=GetOrderStatusArguments,
            handler=_handle_get_order_status,
            agent_access=AgentToolAccess.READ,
        ),
        "cancel_order": RegisteredTool(
            definition=cancel_order_tool_definition(),
            arguments_model=CancelOrderArguments,
            handler=_handle_cancel_order,
            effect=ToolEffect.WRITE,
            agent_access=AgentToolAccess.PROPOSAL,
        ),
    }
)
