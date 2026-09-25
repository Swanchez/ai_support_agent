from dataclasses import dataclass

from ai_support_agent.agents.core import (
    AgentAction,
    AgentActionProposal,
    AgentDecision,
    AgentPlannerResult,
    AgentRunResult,
    AgentRunner,
    AgentState,
    AgentToolRegistry,
    AgentToolResult,
)
from ai_support_agent.llm_client import LlmResult
from ai_support_agent.schemas import AnswerStatus, SupportResponse


def response(answer: str) -> SupportResponse:
    return SupportResponse(
        status=AnswerStatus.ANSWERED,
        answer=answer,
        alternative=None,
        recommendations=[],
        sources=["invented-by-planner"],
    )


def planner_result(decision: AgentDecision) -> AgentPlannerResult:
    return AgentPlannerResult(decision, LlmResult("", "fake-planner", 2, 1, 3))


@dataclass
class StubTool:
    result: AgentToolResult
    received_arguments: list[dict[str, object]]

    def execute(self, arguments: dict[str, object]) -> AgentToolResult:
        self.received_arguments.append(arguments)
        return self.result


class TwoStepPlanner:
    def decide(self, state: AgentState) -> AgentPlannerResult:
        if state.step_count == 0:
            return planner_result(AgentDecision(AgentAction("search_knowledge_base", {"query": "return"})))
        if state.step_count == 1:
            return planner_result(AgentDecision(AgentAction("get_order_status", {"order_id": "ORD-1001"})))
        return planner_result(AgentDecision(response=response("Final answer")))

    def finalize_after_limit(self, state: AgentState) -> AgentPlannerResult:
        _ = state
        return planner_result(AgentDecision(response=response("Limit answer")))


class LoopingPlanner:
    def decide(self, state: AgentState) -> AgentPlannerResult:
        _ = state
        return planner_result(AgentDecision(AgentAction("search_knowledge_base", {"query": "again"})))

    def finalize_after_limit(self, state: AgentState) -> AgentPlannerResult:
        return planner_result(AgentDecision(response=response(f"Stopped after {state.step_count} actions")))


class WriteProposalPlanner:
    def decide(self, state: AgentState) -> AgentPlannerResult:
        _ = state
        return planner_result(
            AgentDecision(proposal=AgentActionProposal("cancel_order", {"order_id": "ORD-1003"}))
        )

    def finalize_after_limit(self, state: AgentState) -> AgentPlannerResult:
        raise AssertionError(f"Should not reach a limit with proposal: {state}")


def test_agent_runs_multiple_actions_and_uses_only_observation_provenance() -> None:
    search = StubTool(
        AgentToolResult({"ok": True, "chunks": []}, ("refund-policy-v1",)),
        [],
    )
    status = StubTool(
        AgentToolResult({"ok": True, "status": "shipped"}, ("get_order_status",)),
        [],
    )
    result = AgentRunner(
        planner=TwoStepPlanner(),
        tools=AgentToolRegistry(
            {"search_knowledge_base": search, "get_order_status": status}
        ),
    ).run("Can I return ORD-1001?")

    assert isinstance(result, AgentRunResult)
    assert result.state.step_count == 2
    assert result.state.total_tokens == 9
    assert result.step_limit_reached is False
    assert result.response.sources == ["refund-policy-v1", "get_order_status"]
    assert search.received_arguments == [{"query": "return"}]
    assert status.received_arguments == [{"order_id": "ORD-1001"}]


def test_agent_stops_tool_loop_at_max_steps_and_still_returns_answer() -> None:
    tool = StubTool(AgentToolResult({"ok": True}, ("source-a",)), [])
    result = AgentRunner(
        planner=LoopingPlanner(),
        tools=AgentToolRegistry({"search_knowledge_base": tool}),
        max_steps=2,
    ).run("Loop forever")

    assert result.state.step_count == 2
    assert result.state.total_tokens == 9
    assert result.step_limit_reached is True
    assert result.response.answer == "Stopped after 2 actions"
    assert tool.received_arguments == [{"query": "again"}, {"query": "again"}]


def test_agent_registry_blocks_unknown_tools() -> None:
    result = AgentToolRegistry({}).execute(AgentAction("delete_everything", {}))

    assert result.data == {"ok": False, "code": "tool_not_allowed"}


def test_agent_returns_a_write_proposal_without_executing_any_tool() -> None:
    cancel_tool = StubTool(AgentToolResult({"ok": True}, ("cancel_order",)), [])

    result = AgentRunner(
        planner=WriteProposalPlanner(),
        tools=AgentToolRegistry({"cancel_order": cancel_tool}),
    ).run("Отмени заказ ORD-1003")

    assert result.response is None
    assert result.proposal == AgentActionProposal("cancel_order", {"order_id": "ORD-1003"})
    assert result.state.step_count == 0
    assert cancel_tool.received_arguments == []
