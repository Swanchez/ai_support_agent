"""Application service that turns safe agent proposals into pending confirmations."""

from dataclasses import dataclass, field
import re

from ai_support_agent.agents.core import (
    AgentConversationMessage,
    AgentRunResult,
    AgentRunner,
    AgentState,
)
from ai_support_agent.schemas import AnswerStatus, SupportResponse
from ai_support_agent.service import AnswerResult
from ai_support_agent.tools.confirmation import InMemoryPendingActionStore
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.executor import ToolExecutor


@dataclass
class AgentAssistantService:
    """Bridge the generic agent loop to application-owned HITL state."""

    agent_runner: AgentRunner
    tool_executor: ToolExecutor
    tool_context: ToolExecutionContext
    pending_action_store: InMemoryPendingActionStore
    last_run: AgentRunResult | None = field(init=False, default=None)

    def answer(
        self,
        user_question: str,
        *,
        history: tuple[AgentConversationMessage, ...] = (),
    ) -> AnswerResult:
        """Return an answer or store an approved-by-policy write proposal for confirmation."""

        run = self.agent_runner.run(user_question, history=history)
        self.last_run = run
        model = run.state.model or "unknown-agent-model"
        if run.proposal is None:
            assert run.response is not None
            return _answer_result(run.response, model, run.state)

        proposal = run.proposal
        if (
            not self.tool_executor.is_registered(proposal.tool_name)
            or not self.tool_executor.requires_confirmation(proposal.tool_name)
        ):
            return _answer_result(
                SupportResponse(
                    status=AnswerStatus.CLARIFICATION_NEEDED,
                    answer="Запрошенное действие недоступно для подтверждения.",
                    alternative=None,
                    recommendations=[],
                    sources=[],
                ),
                model,
                run.state,
            )

        validation = self.tool_executor.validate_arguments(
            proposal.tool_name,
            proposal.arguments,
        )
        if validation.failure is not None:
            return _answer_result(
                SupportResponse(
                    status=AnswerStatus.CLARIFICATION_NEEDED,
                    answer="Не удалось подготовить действие к подтверждению: проверьте данные.",
                    alternative=None,
                    recommendations=[],
                    sources=[],
                ),
                model,
                run.state,
            )
        assert validation.arguments is not None
        validated_arguments = _arguments_as_dict(validation.arguments)
        if (
            proposal.tool_name == "request_return"
            and not _reason_is_explicit(user_question, validated_arguments)
        ):
            return _answer_result(
                SupportResponse(
                    status=AnswerStatus.CLARIFICATION_NEEDED,
                    answer="Укажите, пожалуйста, краткую причину возврата товара.",
                    alternative=None,
                    recommendations=[],
                    sources=[],
                ),
                model,
                run.state,
            )

        self.pending_action_store.create(
            user_id=self.tool_context.current_user_id,
            conversation_id=self.tool_context.conversation_id,
            tool_name=proposal.tool_name,
            arguments=validated_arguments,
        )
        return _answer_result(
            SupportResponse(
                status=AnswerStatus.CONFIRMATION_REQUIRED,
                answer="Запрошенное действие ожидает вашего подтверждения.",
                alternative=None,
                recommendations=[],
                sources=[],
            ),
            model,
            run.state,
        )


def _answer_result(
    response: SupportResponse,
    model: str,
    state: AgentState,
) -> AnswerResult:
    """Map application-owned agent state to the existing console result contract."""

    return AnswerResult(
        response=response,
        model=model,
        input_tokens=state.input_tokens,
        output_tokens=state.output_tokens,
        total_tokens=state.total_tokens,
    )


def _reason_is_explicit(user_question: str, arguments: dict[str, object]) -> bool:
    """Reject an LLM-invented return reason before it reaches confirmation."""

    reason = str(arguments.get("reason", "")).casefold()
    question_words = set(re.findall(r"[а-яёa-z]{4,}", user_question.casefold()))
    reason_words = set(re.findall(r"[а-яёa-z]{4,}", reason))
    ignored = {"заказ", "товар", "вернуть", "возврат", "нужно", "хочу", "причина"}
    return bool((reason_words - ignored) & question_words)


def _arguments_as_dict(arguments: object) -> dict[str, object]:
    """Normalize validated tool arguments without coupling to one implementation."""

    if isinstance(arguments, dict):
        return arguments
    return arguments.model_dump(mode="json")  # type: ignore[union-attr]
