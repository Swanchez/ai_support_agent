from dataclasses import dataclass

from ai_support_agent.agents.tools import (
    MyOrdersAgentTool,
    OrderStatusAgentTool,
    SearchKnowledgeBaseTool,
    agent_tool_definitions,
    proposal_agent_tool_names,
    read_agent_tool_definitions,
)
from ai_support_agent.rag.models import SourceType
from ai_support_agent.rag.retriever import RetrievedChunk
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR


@dataclass
class StubRetriever:
    chunks: list[RetrievedChunk]
    received_query: str | None = None

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        _ = top_k, threshold, max_chunks_per_document
        self.received_query = question
        return self.chunks


def test_search_knowledge_base_returns_compact_evidence_and_provenance() -> None:
    retriever = StubRetriever(
        [
            RetrievedChunk(
                "Return timing",
                0.9,
                "refund-policy-v1",
                "Refund policy",
                source_type=SourceType.INTERNAL_POLICY,
            )
        ]
    )

    result = SearchKnowledgeBaseTool(retriever).execute({"query": "When is refund?"})

    assert retriever.received_query == "When is refund?"
    assert result.data["chunks"] == [
        {
            "text": "Return timing",
            "source_id": "refund-policy-v1",
            "source_type": "internal_policy",
            "page_number": None,
        }
    ]
    assert result.source_ids == ("refund-policy-v1",)


def test_search_knowledge_base_rejects_invalid_arguments_without_retrieval() -> None:
    retriever = StubRetriever([])

    result = SearchKnowledgeBaseTool(retriever).execute({"query": "", "top_k": 999})

    assert result.data == {"ok": False, "code": "invalid_tool_arguments"}
    assert retriever.received_query is None


def test_order_status_adapter_reuses_existing_executor_and_ownership_check() -> None:
    tool = OrderStatusAgentTool(DEFAULT_TOOL_EXECUTOR, DEMO_TOOL_CONTEXT)

    own_order = tool.execute({"order_id": "ORD-1001"})
    other_users_order = tool.execute({"order_id": "ORD-1002"})

    assert own_order.data["order"]["status"] == "shipped"
    assert own_order.source_ids == ("get_order_status",)
    assert other_users_order.data["code"] == "order_not_found"


def test_my_orders_adapter_reuses_the_ownership_scoped_executor() -> None:
    tool = MyOrdersAgentTool(DEFAULT_TOOL_EXECUTOR, DEMO_TOOL_CONTEXT)

    result = tool.execute({})

    assert [order["order_id"] for order in result.data["orders"]] == [
        "ORD-1001",
        "ORD-1003",
        "ORD-1004",
    ]
    assert result.source_ids == ("get_my_orders",)


def test_agent_tool_definitions_expose_only_read_actions() -> None:
    assert [definition["name"] for definition in read_agent_tool_definitions()] == [
        "search_knowledge_base",
        "get_my_orders",
        "get_order_status",
    ]


def test_agent_tool_definitions_expose_writes_only_as_proposals() -> None:
    assert [definition["name"] for definition in agent_tool_definitions()] == [
        "search_knowledge_base",
        "get_my_orders",
        "get_order_status",
        "cancel_order",
        "request_return",
    ]
    assert proposal_agent_tool_names() == frozenset({"cancel_order", "request_return"})
