import asyncio
from dataclasses import dataclass

from mcp import Client

from ai_support_agent.mcp_server import (
    SERVICE_OVERVIEW,
    SERVICE_OVERVIEW_URI,
    SUPPORT_POLICY_ANSWER_PROMPT,
    create_authenticated_support_mcp_server,
    create_support_mcp_server,
)
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.rag.models import SourceType
from ai_support_agent.rag.retriever import RetrievedChunk


@dataclass
class StubRetriever:
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
        return [
            RetrievedChunk(
                text="Возврат обрабатывается после поступления товара на склад.",
                score=0.91,
                document_id="refund-policy-v1",
                title="Refund policy",
                source_type=SourceType.INTERNAL_POLICY,
            )
        ]


def test_mcp_server_discovers_and_calls_rag_tool_in_memory() -> None:
    retriever = StubRetriever()

    async def run() -> None:
        async with Client(create_support_mcp_server(retriever)) as client:
            tools = await client.list_tools()
            result = await client.call_tool(
                "search_knowledge_base",
                {"query": "Когда поступят деньги за возврат?"},
            )

        assert [tool.name for tool in tools.tools] == ["search_knowledge_base"]
        assert result.is_error is False
        assert result.structured_content == {
            "context": "Возврат обрабатывается после поступления товара на склад.",
            "chunks": [
                {
                    "text": "Возврат обрабатывается после поступления товара на склад.",
                    "source_id": "refund-policy-v1",
                    "source_type": "internal_policy",
                    "page_number": None,
                    "score": 0.91,
                }
            ]
        }

    asyncio.run(run())
    assert retriever.received_query == "Когда поступят деньги за возврат?"


def test_mcp_server_discovers_and_reads_static_overview_resource() -> None:
    async def run() -> None:
        async with Client(create_support_mcp_server(StubRetriever())) as client:
            resources = await client.list_resources()
            result = await client.read_resource(SERVICE_OVERVIEW_URI)

        assert [str(resource.uri) for resource in resources.resources] == [
            SERVICE_OVERVIEW_URI
        ]
        assert result.contents[0].text == SERVICE_OVERVIEW

    asyncio.run(run())


def test_mcp_server_discovers_and_renders_policy_answer_prompt() -> None:
    async def run() -> None:
        async with Client(create_support_mcp_server(StubRetriever())) as client:
            prompts = await client.list_prompts()
            result = await client.get_prompt(
                SUPPORT_POLICY_ANSWER_PROMPT,
                {"question": "Когда придут деньги за возврат?"},
            )

        assert [prompt.name for prompt in prompts.prompts] == [
            SUPPORT_POLICY_ANSWER_PROMPT
        ]
        assert result.messages[0].content.text == (
            "Найди сведения по вопросу ниже с помощью tool "
            "search_knowledge_base. Затем дай краткий ответ только по найденным "
            "источникам. Если источников недостаточно, честно сообщи об этом.\n\n"
            "Вопрос: Когда придут деньги за возврат?"
        )

    asyncio.run(run())


def test_authenticated_mcp_server_returns_status_only_for_context_owner() -> None:
    async def run() -> None:
        server = create_authenticated_support_mcp_server(
            StubRetriever(),
            ToolExecutionContext(current_user_id="demo-user-1"),
        )
        async with Client(server) as client:
            tools = await client.list_tools()
            owned = await client.call_tool("get_order_status", {"order_id": "ORD-1001"})
            foreign = await client.call_tool("get_order_status", {"order_id": "ORD-1002"})

        assert [tool.name for tool in tools.tools] == [
            "search_knowledge_base",
            "get_order_status",
        ]
        assert owned.structured_content == {
            "ok": True,
            "order": {
                "order_id": "ORD-1001",
                "status": "shipped",
                "updated_at": "2026-09-20",
                "estimated_delivery": "2026-09-24",
            },
        }
        assert foreign.structured_content == {
            "ok": False,
            "code": "order_not_found",
            "message": "Order was not found.",
            "retryable": False,
        }

    asyncio.run(run())


def test_mcp_server_includes_diagnostics_only_when_requested() -> None:
    retriever = StubRetriever()

    async def run() -> None:
        async with Client(create_support_mcp_server(retriever)) as client:
            result = await client.call_tool(
                "search_knowledge_base",
                {"query": "Тестовый запрос", "debug": True},
            )

        assert result.structured_content is not None
        assert result.structured_content["debug"] == {
            "received_query": "Тестовый запрос",
            "retrieved_chunk_count": 1,
            "threshold": 0.7,
        }

    asyncio.run(run())
