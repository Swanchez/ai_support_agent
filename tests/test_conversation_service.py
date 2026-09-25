from dataclasses import dataclass

from pydantic import BaseModel

from ai_support_agent.conversation_service import ConversationService
from ai_support_agent.llm_client import LlmResult
from ai_support_agent.tools.confirmation import InMemoryPendingActionStore, PendingToolAction
from ai_support_agent.tools.confirmation_resolver import (
    ConfirmationDecision,
    ConfirmationResolution,
    ConfirmationResolutionResult,
)
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT, ToolExecutionContext
from ai_support_agent.tools.audit import InMemoryAuditSink
from ai_support_agent.tools.executor import RegisteredTool, ToolEffect, ToolExecutor
from ai_support_agent.tools.order_status import (
    CancelOrderArguments,
    InMemoryOrderRepository,
    OrderStatus,
    OrderStatusData,
    StoredOrder,
    cancel_order,
)


@dataclass
class StubAssistantService:
    pending_action_store: InMemoryPendingActionStore
    tool_context: ToolExecutionContext
    tool_executor: ToolExecutor


@dataclass
class StubConfirmationResolver:
    decision: ConfirmationDecision

    def resolve(
        self,
        user_reply: str,
        pending_action: PendingToolAction,
    ) -> ConfirmationResolutionResult:
        _ = user_reply, pending_action
        return ConfirmationResolutionResult(
            resolution=ConfirmationResolution(decision=self.decision),
            llm_result=LlmResult("", "resolver", 4, 1, 5),
        )


def create_cancel_executor(repository: InMemoryOrderRepository) -> ToolExecutor:
    def handler(arguments: BaseModel, context: ToolExecutionContext) -> BaseModel:
        return cancel_order(
            CancelOrderArguments.model_validate(arguments),
            current_user_id=context.current_user_id,
            idempotency_key=context.idempotency_key or "missing-test-key",
            repository=repository,
        )

    return ToolExecutor(
        {
            "cancel_order": RegisteredTool(
                definition={},
                arguments_model=CancelOrderArguments,
                handler=handler,
                effect=ToolEffect.WRITE,
            )
        }
    )


def create_service(
    decision: ConfirmationDecision,
    audit_sink: InMemoryAuditSink | None = None,
) -> tuple[ConversationService, InMemoryOrderRepository]:
    repository = InMemoryOrderRepository(
        {
            "ORD-1003": StoredOrder(
                "demo-user-1",
                OrderStatusData(
                    order_id="ORD-1003",
                    status=OrderStatus.PACKED,
                    updated_at="2026-09-22",
                ),
            )
        }
    )
    pending_store = InMemoryPendingActionStore(audit_sink=audit_sink or InMemoryAuditSink())
    pending_store.create(
        user_id="demo-user-1",
        tool_name="cancel_order",
        arguments={"order_id": "ORD-1003"},
    )
    return (
        ConversationService(
            assistant_service=StubAssistantService(
                pending_action_store=pending_store,
                tool_context=DEMO_TOOL_CONTEXT,
                tool_executor=create_cancel_executor(repository),
            ),
            confirmation_resolver=StubConfirmationResolver(decision),
        ),
        repository,
    )


def test_confirmation_executes_only_the_saved_action_after_semantic_confirm() -> None:
    service, repository = create_service(ConfirmationDecision.CONFIRM)

    assert service.has_pending_action() is True

    result = service.answer("Да, отменяй")

    assert result.response.answer == "Заказ ORD-1003 отменён."
    assert result.response.sources == ["cancel_order"]
    assert repository.find_visible_to("ORD-1003", "demo-user-1").status is OrderStatus.CANCELLED
    assert service.assistant_service.pending_action_store.find_for_user("demo-user-1") is None
    assert service.has_pending_action() is False


def test_rejection_discards_pending_action_without_executing_it() -> None:
    service, repository = create_service(ConfirmationDecision.REJECT)

    result = service.answer("Нет, передумал")

    assert result.response.answer == "Действие отменено и не было выполнено."
    assert repository.find_visible_to("ORD-1003", "demo-user-1").status is OrderStatus.PACKED
    assert service.assistant_service.pending_action_store.find_for_user("demo-user-1") is None


def test_unclear_reply_keeps_pending_action_unchanged() -> None:
    audit_sink = InMemoryAuditSink()
    service, repository = create_service(ConfirmationDecision.UNCLEAR, audit_sink)

    result = service.answer("А когда его привезут?")

    assert result.response.status.value == "clarification_needed"
    assert repository.find_visible_to("ORD-1003", "demo-user-1").status is OrderStatus.PACKED
    assert service.assistant_service.pending_action_store.find_for_user("demo-user-1") is not None
    assert audit_sink.events[-1].outcome == "confirmation_unclear"
