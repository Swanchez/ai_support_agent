from dataclasses import dataclass

from ai_support_agent.assistant_service import AssistantService, build_assistant_tool_call
from ai_support_agent.llm_client import LlmResult
from ai_support_agent.rag.retriever import RetrievedChunk
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR, ToolExecutor
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT, ToolExecutionContext
from ai_support_agent.tools.confirmation import InMemoryPendingActionStore
from ai_support_agent.tools.gemini_adapter import ToolCall
from ai_support_agent.tools.gemini_tool_client import ToolCallingResult


@dataclass
class StubRetriever:
    chunks: list[RetrievedChunk]

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        _ = question, top_k, threshold, max_chunks_per_document
        return self.chunks


@dataclass
class StubToolClient:
    result: ToolCallingResult
    received_call: object | None = None
    received_executor: ToolExecutor | None = None
    received_context: ToolExecutionContext | None = None

    def complete_with_tools(
        self,
        call: object,
        executor: ToolExecutor,
        context: ToolExecutionContext,
    ) -> ToolCallingResult:
        self.received_call = call
        self.received_executor = executor
        self.received_context = context
        return self.result


def tool_result(*, sources: tuple[str, ...]) -> ToolCallingResult:
    return ToolCallingResult(
        llm_result=LlmResult(
            text=(
                '{"status":"answered","answer":"Готово",'
                '"alternative":null,"recommendations":[],"sources":["invented"]}'
            ),
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        ),
        executed_tool_names=sources,
    )


def test_assistant_service_combines_only_trusted_rag_and_tool_sources() -> None:
    client = StubToolClient(tool_result(sources=("get_order_status",)))
    service = AssistantService(
        retriever=StubRetriever(
            [
                RetrievedChunk("Return rule", 0.9, "refund-policy-v1", "Refund policy"),
                RetrievedChunk("Duplicate", 0.8, "refund-policy-v1", "Refund policy"),
            ]
        ),
        tool_client=client,
        tool_executor=DEFAULT_TOOL_EXECUTOR,
        tool_context=DEMO_TOOL_CONTEXT,
        pending_action_store=InMemoryPendingActionStore(),
    )

    result = service.answer("What is the status of ORD-1001 and the refund deadline?")

    assert result.response.sources == ["refund-policy-v1", "get_order_status"]
    assert result.total_tokens == 15
    assert client.received_executor is DEFAULT_TOOL_EXECUTOR
    assert client.received_context is DEMO_TOOL_CONTEXT


def test_assistant_tool_call_includes_rag_context_and_narrow_tool_rules() -> None:
    call = build_assistant_tool_call("Where is ORD-1001?", "RAG evidence")

    assert "RAG evidence" in call.messages[1]["content"]
    assert "get_order_status" in call.messages[0]["content"]
    assert call.messages[2]["content"] == "Where is ORD-1001?"


def test_assistant_service_keeps_rag_sources_when_no_tool_was_executed() -> None:
    service = AssistantService(
        retriever=StubRetriever(
            [RetrievedChunk("Return rule", 0.9, "refund-policy-v1", "Refund policy")]
        ),
        tool_client=StubToolClient(tool_result(sources=())),
        tool_executor=DEFAULT_TOOL_EXECUTOR,
        tool_context=DEMO_TOOL_CONTEXT,
        pending_action_store=InMemoryPendingActionStore(),
    )

    result = service.answer("What are the return terms?")

    assert result.response.sources == ["refund-policy-v1"]


def test_assistant_service_saves_a_write_call_as_pending_for_current_user() -> None:
    pending_store = InMemoryPendingActionStore()
    client = StubToolClient(
        ToolCallingResult(
            llm_result=LlmResult(
                text=(
                    '{"status":"clarification_needed","answer":"Подтвердите отмену.",'
                    '"alternative":null,"recommendations":[],"sources":[]}'
                ),
                model="test-model",
                input_tokens=10,
                output_tokens=5,
                total_tokens=15,
            ),
            executed_tool_names=(),
            pending_tool_calls=(
                ToolCall("call-1", "cancel_order", {"order_id": "ORD-1001"}),
            ),
        )
    )
    service = AssistantService(
        retriever=StubRetriever([]),
        tool_client=client,
        tool_executor=DEFAULT_TOOL_EXECUTOR,
        tool_context=DEMO_TOOL_CONTEXT,
        pending_action_store=pending_store,
    )

    result = service.answer("Cancel ORD-1001")
    pending = next(iter(pending_store._actions.values()))

    assert result.response.answer == "Запрошенное действие ожидает вашего подтверждения."
    assert pending.user_id == "demo-user-1"
    assert pending.tool_name == "cancel_order"
    assert pending.arguments == {"order_id": "ORD-1001"}
