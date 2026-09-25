"""Provider-neutral, bounded agent loop for read-only observations."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from ai_support_agent.llm_client import LlmResult
from ai_support_agent.schemas import SupportResponse


@dataclass(frozen=True)
class AgentAction:
    """One model-proposed action, still subject to the application's registry."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentActionProposal:
    """A requested write action that still needs application and user approval."""

    tool_name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentDecision:
    """A planner may execute a read action, propose a write, or return a response."""

    action: AgentAction | None = None
    response: SupportResponse | None = None
    proposal: AgentActionProposal | None = None

    def __post_init__(self) -> None:
        if sum(value is not None for value in (self.action, self.response, self.proposal)) != 1:
            raise ValueError("Agent decision must contain exactly one action, proposal, or response.")


@dataclass(frozen=True)
class AgentToolResult:
    """Structured observation and provenance returned by one application tool."""

    data: dict[str, Any]
    source_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentObservation:
    """One immutable record visible to subsequent planner steps."""

    action: AgentAction
    result: AgentToolResult


@dataclass(frozen=True)
class AgentPlannerResult:
    """One planner decision together with provider-reported token usage."""

    decision: AgentDecision
    llm_result: LlmResult


@dataclass
class AgentState:
    """Application-owned state for one bounded agent run."""

    original_question: str
    observations: list[AgentObservation] = field(default_factory=list)
    step_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model: str | None = None


class AgentTool(Protocol):
    """Read-only agent action with structured, provenance-aware output."""

    def execute(self, arguments: dict[str, Any]) -> AgentToolResult:
        """Execute validated application logic, never model-provided code."""


class AgentPlanner(Protocol):
    """Boundary implemented later by a provider-specific LLM planner."""

    def decide(self, state: AgentState) -> AgentPlannerResult:
        """Choose the next bounded action or finish using current observations."""

    def finalize_after_limit(self, state: AgentState) -> AgentPlannerResult:
        """Return the best honest answer when the action budget is exhausted."""


@dataclass(frozen=True)
class AgentToolRegistry:
    """Explicit whitelist for all actions available to the planner."""

    tools: Mapping[str, AgentTool]

    def execute(self, action: AgentAction) -> AgentToolResult:
        """Run only an allowed tool; unknown names become safe observations."""

        tool = self.tools.get(action.name)
        if tool is None:
            return AgentToolResult(
                data={"ok": False, "code": "tool_not_allowed"},
            )
        return tool.execute(action.arguments)


@dataclass(frozen=True)
class AgentRunResult:
    """A final response or a pending write proposal with execution state."""

    response: SupportResponse | None
    state: AgentState
    step_limit_reached: bool
    proposal: AgentActionProposal | None = None


@dataclass
class AgentRunner:
    """Runs planner actions under a strict application-owned action budget."""

    planner: AgentPlanner
    tools: AgentToolRegistry
    max_steps: int = 3

    def run(self, user_question: str) -> AgentRunResult:
        """Execute at most max_steps actions, then force one final answer."""

        state = AgentState(original_question=user_question)
        for _ in range(self.max_steps):
            planner_result = self.planner.decide(state)
            _add_usage(state, planner_result.llm_result)
            decision = planner_result.decision
            if decision.response is not None:
                return AgentRunResult(
                    response=_attach_observation_sources(decision.response, state),
                    state=state,
                    step_limit_reached=False,
                )
            if decision.proposal is not None:
                return AgentRunResult(
                    response=None,
                    state=state,
                    step_limit_reached=False,
                    proposal=decision.proposal,
                )
            action = decision.action
            assert action is not None
            state.observations.append(AgentObservation(action, self.tools.execute(action)))
            state.step_count += 1

        final_response = self.planner.finalize_after_limit(state)
        _add_usage(state, final_response.llm_result)
        return AgentRunResult(
            response=_attach_observation_sources(final_response.decision.response, state),
            state=state,
            step_limit_reached=True,
        )


def _attach_observation_sources(response: SupportResponse, state: AgentState) -> SupportResponse:
    """Derive provenance from executed tools, never from planner-produced JSON."""

    source_ids = [
        source_id
        for observation in state.observations
        for source_id in observation.result.source_ids
    ]
    return response.model_copy(update={"sources": list(dict.fromkeys(source_ids))})


def _add_usage(state: AgentState, result: LlmResult) -> None:
    """Accumulate every planner request so the agent's total cost is observable."""

    state.input_tokens += result.input_tokens
    state.output_tokens += result.output_tokens
    state.total_tokens += result.total_tokens
    state.model = result.model
