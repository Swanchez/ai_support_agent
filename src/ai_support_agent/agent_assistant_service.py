"""Application service that turns safe agent proposals into pending confirmations."""

from dataclasses import dataclass, field

from ai_support_agent.agents.core import AgentRunResult, AgentRunner, AgentState
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

    def answer(self, user_question: str) -> AnswerResult:
        """Return an answer or store an approved-by-policy write proposal for confirmation."""

        run = self.agent_runner.run(user_question)
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

        self.pending_action_store.create(
            user_id=self.tool_context.current_user_id,
            tool_name=proposal.tool_name,
            arguments=validation.arguments.model_dump(mode="json"),
        )
        return _answer_result(
            SupportResponse(
                status=AnswerStatus.CLARIFICATION_NEEDED,
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
