from dataclasses import dataclass

from ai_support_agent.agents.core import AgentDecision, AgentPlannerResult, AgentState
from ai_support_agent.agents.runtime import AGENT_MAX_STEPS, build_agent_runner
from ai_support_agent.llm_client import LlmResult
from ai_support_agent.rag.retriever import RetrievedChunk
from ai_support_agent.schemas import AnswerStatus, SupportResponse
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR


@dataclass
class StubRetriever:
    def retrieve(self, *args: object, **kwargs: object) -> list[RetrievedChunk]:
        _ = args, kwargs
        return []


class FinalPlanner:
    def decide(self, state: AgentState) -> AgentPlannerResult:
        _ = state
        return self._result()

    def finalize_after_limit(self, state: AgentState) -> AgentPlannerResult:
        _ = state
        return self._result()

    @staticmethod
    def _result() -> AgentPlannerResult:
        return AgentPlannerResult(
            AgentDecision(
                response=SupportResponse(
                    status=AnswerStatus.INSUFFICIENT_CONTEXT,
                    answer="No evidence.",
                    alternative=None,
                    recommendations=[],
                    sources=[],
                )
            ),
            LlmResult("", "fake", 1, 1, 2),
        )


def test_agent_runtime_wires_only_read_tools_and_shared_safeguards() -> None:
    runner = build_agent_runner(
        retriever=StubRetriever(),
        planner=FinalPlanner(),
        executor=DEFAULT_TOOL_EXECUTOR,
        context=DEMO_TOOL_CONTEXT,
    )

    assert runner.max_steps == AGENT_MAX_STEPS
    assert list(runner.tools.tools) == [
        "search_knowledge_base",
        "get_my_orders",
        "get_order_status",
    ]
    assert runner.run("Question").response.answer == "No evidence."
