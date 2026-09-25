from types import SimpleNamespace

from ai_support_agent.agents.core import (
    AgentAction,
    AgentActionProposal,
    AgentObservation,
    AgentState,
    AgentToolResult,
)
from ai_support_agent.agents.gemini_planner import AGENT_SYSTEM_PROMPT, GeminiAgentPlanner
from ai_support_agent.config import GeminiConfig
from ai_support_agent.llm_client import gemini_response_format


class FakeInteractionsApi:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return self.responses.pop(0)


def response(*, output_text: str, steps: list[object]) -> object:
    return SimpleNamespace(
        output_text=output_text,
        steps=steps,
        usage=SimpleNamespace(total_input_tokens=10, total_output_tokens=4, total_tokens=14),
    )


def test_gemini_agent_planner_translates_one_function_call_to_agent_action() -> None:
    interactions = FakeInteractionsApi(
        [
            response(
                output_text="",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="call-1",
                        name="search_knowledge_base",
                        arguments={"query": "refund"},
                    )
                ],
            )
        ]
    )
    planner = GeminiAgentPlanner(
        GeminiConfig(api_key="test-key", model="test-model"),
        tool_definitions=[{"name": "search_knowledge_base"}],
        sdk_client=SimpleNamespace(interactions=interactions),
    )

    result = planner.decide(AgentState("When is refund?"))

    assert result.decision.action == AgentAction("search_knowledge_base", {"query": "refund"})
    assert result.llm_result.total_tokens == 14
    assert interactions.calls[0]["tools"] == [{"name": "search_knowledge_base"}]
    assert interactions.calls[0]["response_format"] == gemini_response_format()
    assert interactions.calls[0]["store"] is False


def test_gemini_agent_planner_translates_proposal_only_function_to_proposal() -> None:
    interactions = FakeInteractionsApi(
        [
            response(
                output_text="",
                steps=[
                    SimpleNamespace(
                        type="function_call",
                        id="call-1",
                        name="cancel_order",
                        arguments={"order_id": "ORD-1003"},
                    )
                ],
            )
        ]
    )
    planner = GeminiAgentPlanner(
        GeminiConfig(api_key="test-key", model="test-model"),
        tool_definitions=[{"name": "cancel_order"}],
        proposal_tool_names=frozenset({"cancel_order"}),
        sdk_client=SimpleNamespace(interactions=interactions),
    )

    result = planner.decide(AgentState("Cancel ORD-1003"))

    assert result.decision.proposal == AgentActionProposal(
        "cancel_order", {"order_id": "ORD-1003"}
    )
    assert result.decision.action is None


def test_gemini_agent_planner_forces_final_json_without_tools_after_limit() -> None:
    interactions = FakeInteractionsApi(
        [
            response(
                output_text=(
                    '{"status":"answered","answer":"Enough evidence",'
                    '"alternative":null,"recommendations":[],"sources":["invented"]}'
                ),
                steps=[],
            )
        ]
    )
    planner = GeminiAgentPlanner(
        GeminiConfig(api_key="test-key", model="test-model"),
        tool_definitions=[{"name": "search_knowledge_base"}],
        sdk_client=SimpleNamespace(interactions=interactions),
    )
    state = AgentState(
        "When is refund?",
        observations=[
            AgentObservation(
                AgentAction("search_knowledge_base", {"query": "refund"}),
                AgentToolResult({"ok": True, "chunks": []}, ("refund-policy-v1",)),
            )
        ],
        step_count=3,
    )

    result = planner.finalize_after_limit(state)

    assert result.decision.response.answer == "Enough evidence"
    assert interactions.calls[0]["tools"] == []
    assert "Лимит действий исчерпан" in interactions.calls[0]["system_instruction"]


def test_agent_prompt_instructs_the_model_not_to_retry_known_tool_errors() -> None:
    assert "result.ok=false" in AGENT_SYSTEM_PROMPT
    assert "order_not_found" in AGENT_SYSTEM_PROMPT
    assert "Не повторяй тот же инструмент" in AGENT_SYSTEM_PROMPT


def test_agent_prompt_uses_a_generic_proposal_only_rule() -> None:
    assert "proposal-only" in AGENT_SYSTEM_PROMPT
    assert "cancel_order является proposal-only" not in AGENT_SYSTEM_PROMPT
