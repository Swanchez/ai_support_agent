from ai_support_agent.agents.core import AgentAction, AgentObservation, AgentState, AgentToolResult
from ai_support_agent.agents.trace import build_agent_trace, format_agent_trace


def test_trace_summarises_actions_without_document_contents() -> None:
    state = AgentState(
        original_question="Где мой заказ?",
        observations=[
            AgentObservation(
                action=AgentAction("search_knowledge_base", {"query": "возврат"}),
                result=AgentToolResult(
                    {"ok": True, "chunks": [{"text": "длинный фрагмент"}]},
                    ("refund-policy-v1",),
                ),
            )
        ],
    )

    trace = build_agent_trace(state)

    assert trace[0].action_name == "search_knowledge_base"
    assert trace[0].chunk_count == 1
    assert trace[0].source_ids == ("refund-policy-v1",)
    assert "длинный фрагмент" not in format_agent_trace(state)


def test_trace_redacts_common_secret_argument_names() -> None:
    state = AgentState(
        original_question="test",
        observations=[
            AgentObservation(
                action=AgentAction("test_tool", {"api_key": "not-for-console"}),
                result=AgentToolResult({"ok": False, "code": "tool_failed"}),
            )
        ],
    )

    trace = format_agent_trace(state)

    assert "[redacted]" in trace
    assert "not-for-console" not in trace
