from dataclasses import dataclass

from ai_support_agent.agent_assistant_service import AgentAssistantService
from ai_support_agent.agents.core import (
    AgentActionProposal,
    AgentConversationMessage,
    AgentRunResult,
    AgentState,
)
from ai_support_agent.schemas import AnswerStatus, SupportResponse
from ai_support_agent.tools.confirmation import InMemoryPendingActionStore
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR


@dataclass
class StubAgentRunner:
    result: AgentRunResult

    def run(
        self,
        user_question: str,
        *,
        history: tuple[AgentConversationMessage, ...] = (),
    ) -> AgentRunResult:
        _ = user_question, history
        return self.result


def test_agent_proposal_becomes_pending_action_without_executing_write_tool() -> None:
    service = AgentAssistantService(
        agent_runner=StubAgentRunner(
            AgentRunResult(
                response=None,
                state=AgentState(
                    "Отмени заказ ORD-1003",
                    input_tokens=10,
                    output_tokens=2,
                    total_tokens=12,
                    model="fake-agent",
                ),
                step_limit_reached=False,
                proposal=AgentActionProposal("cancel_order", {"order_id": "ORD-1003"}),
            )
        ),
        tool_executor=DEFAULT_TOOL_EXECUTOR,
        tool_context=DEMO_TOOL_CONTEXT,
        pending_action_store=InMemoryPendingActionStore(),
    )

    result = service.answer("Отмени заказ ORD-1003")
    pending = service.pending_action_store.find_for_user("demo-user-1")

    assert result.response.status is AnswerStatus.CONFIRMATION_REQUIRED
    assert result.response.recommendations == []
    assert result.model == "fake-agent"
    assert pending is not None
    assert pending.tool_name == "cancel_order"
    assert pending.arguments == {"order_id": "ORD-1003"}


def test_agent_final_response_uses_existing_answer_result_contract() -> None:
    response = SupportResponse(
        status=AnswerStatus.ANSWERED,
        answer="Данные найдены.",
        alternative=None,
        recommendations=[],
        sources=["refund-policy-v1"],
    )
    service = AgentAssistantService(
        agent_runner=StubAgentRunner(
            AgentRunResult(
                response=response,
                state=AgentState(
                    "Вопрос",
                    input_tokens=7,
                    output_tokens=3,
                    total_tokens=10,
                    model="fake-agent",
                ),
                step_limit_reached=False,
            )
        ),
        tool_executor=DEFAULT_TOOL_EXECUTOR,
        tool_context=DEMO_TOOL_CONTEXT,
        pending_action_store=InMemoryPendingActionStore(),
    )

    result = service.answer("Вопрос")

    assert result.response == response
    assert result.input_tokens == 7
    assert result.total_tokens == 10


def test_invalid_agent_write_proposal_never_becomes_pending_action() -> None:
    service = AgentAssistantService(
        agent_runner=StubAgentRunner(
            AgentRunResult(
                response=None,
                state=AgentState("Отмени", model="fake-agent"),
                step_limit_reached=False,
                proposal=AgentActionProposal("cancel_order", {"order_id": "invalid"}),
            )
        ),
        tool_executor=DEFAULT_TOOL_EXECUTOR,
        tool_context=DEMO_TOOL_CONTEXT,
        pending_action_store=InMemoryPendingActionStore(),
    )

    result = service.answer("Отмени")

    assert result.response.status is AnswerStatus.CLARIFICATION_NEEDED
    assert service.pending_action_store.find_for_user("demo-user-1") is None


def test_agent_cannot_turn_a_read_tool_into_a_confirmation_proposal() -> None:
    service = AgentAssistantService(
        agent_runner=StubAgentRunner(
            AgentRunResult(
                response=None,
                state=AgentState("Статус", model="fake-agent"),
                step_limit_reached=False,
                proposal=AgentActionProposal("get_order_status", {"order_id": "ORD-1001"}),
            )
        ),
        tool_executor=DEFAULT_TOOL_EXECUTOR,
        tool_context=DEMO_TOOL_CONTEXT,
        pending_action_store=InMemoryPendingActionStore(),
    )

    result = service.answer("Статус")

    assert result.response.status is AnswerStatus.CLARIFICATION_NEEDED
    assert service.pending_action_store.find_for_user("demo-user-1") is None
