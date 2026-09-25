"""Minimal stateful dialog flow for pending write-tool confirmations."""

from dataclasses import dataclass, replace
from typing import Any, Protocol

from ai_support_agent.service import AnswerResult
from ai_support_agent.schemas import AnswerStatus, SupportResponse
from ai_support_agent.tools.confirmation_resolver import (
    ConfirmationDecision,
    ConfirmationResolver,
)


class ConfirmableAssistantService(Protocol):
    """Minimal application boundary needed by the shared confirmation workflow."""

    tool_context: Any
    pending_action_store: Any
    tool_executor: Any

    def answer(self, user_question: str) -> AnswerResult:
        """Handle one non-confirmation user request."""


@dataclass
class ConversationService:
    """Routes a message either to the assistant or to its pending confirmation flow."""

    assistant_service: ConfirmableAssistantService
    confirmation_resolver: ConfirmationResolver

    def has_pending_action(self) -> bool:
        """Report whether the current user must answer a confirmation request."""

        return (
            self.assistant_service.pending_action_store.find_for_user(
                self.assistant_service.tool_context.current_user_id
            )
            is not None
        )

    def answer(self, user_message: str) -> AnswerResult:
        """Process a new request or resolve the current user's one pending action."""

        pending = self.assistant_service.pending_action_store.find_for_user(
            self.assistant_service.tool_context.current_user_id
        )
        if pending is None:
            return self.assistant_service.answer(user_message)

        resolution_result = self.confirmation_resolver.resolve(user_message, pending)
        decision = resolution_result.resolution.decision
        if decision is ConfirmationDecision.CONFIRM:
            approved = self.assistant_service.pending_action_store.approve_for_user(
                confirmation_id=pending.confirmation_id,
                user_id=self.assistant_service.tool_context.current_user_id,
            )
            if approved is None:
                return _answer_result(
                    status=AnswerStatus.CLARIFICATION_NEEDED,
                    answer="Подтверждение устарело. Повторите запрос на отмену.",
                    model=resolution_result.llm_result.model,
                    input_tokens=resolution_result.llm_result.input_tokens,
                    output_tokens=resolution_result.llm_result.output_tokens,
                    total_tokens=resolution_result.llm_result.total_tokens,
                )
            result = self.assistant_service.tool_executor.execute(
                approved.tool_name,
                approved.arguments,
                replace(
                    self.assistant_service.tool_context,
                    idempotency_key=approved.confirmation_id,
                ),
                confirmation_granted=True,
            )
            return _result_from_confirmed_tool(
                tool_name=approved.tool_name,
                result=result,
                model=resolution_result.llm_result.model,
                input_tokens=resolution_result.llm_result.input_tokens,
                output_tokens=resolution_result.llm_result.output_tokens,
                total_tokens=resolution_result.llm_result.total_tokens,
            )

        if decision is ConfirmationDecision.REJECT:
            self.assistant_service.pending_action_store.discard_for_user(
                confirmation_id=pending.confirmation_id,
                user_id=self.assistant_service.tool_context.current_user_id,
            )
            return _answer_result(
                status=AnswerStatus.ANSWERED,
                answer="Действие отменено и не было выполнено.",
                model=resolution_result.llm_result.model,
                input_tokens=resolution_result.llm_result.input_tokens,
                output_tokens=resolution_result.llm_result.output_tokens,
                total_tokens=resolution_result.llm_result.total_tokens,
            )

        self.assistant_service.pending_action_store.record_unclear_confirmation(pending)
        return _answer_result(
            status=AnswerStatus.CLARIFICATION_NEEDED,
            answer="Подтвердите или отмените ранее запрошенное действие.",
            model=resolution_result.llm_result.model,
            input_tokens=resolution_result.llm_result.input_tokens,
            output_tokens=resolution_result.llm_result.output_tokens,
            total_tokens=resolution_result.llm_result.total_tokens,
        )


def _result_from_confirmed_tool(
    *,
    tool_name: str,
    result: dict[str, Any],
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> AnswerResult:
    """Render deterministic user-facing text from a confirmed tool result."""

    if result.get("ok") is True and tool_name == "cancel_order":
        order_id = result["order"]["order_id"]
        return _answer_result(
            status=AnswerStatus.ANSWERED,
            answer=f"Заказ {order_id} отменён.",
            sources=[tool_name],
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    return _answer_result(
        status=AnswerStatus.ANSWERED,
        answer="Не удалось выполнить подтверждённое действие.",
        sources=[tool_name],
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _answer_result(
    *,
    status: AnswerStatus,
    answer: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    sources: list[str] | None = None,
) -> AnswerResult:
    """Build a contract-valid answer for application-owned dialog transitions."""

    return AnswerResult(
        response=SupportResponse(
            status=status,
            answer=answer,
            alternative=None,
            recommendations=[],
            sources=sources or [],
        ),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )
