"""Read-only Support MCP server backed by the existing RAG retrieval pipeline."""

from typing import Annotated, Any

from mcp.server import MCPServer
from pydantic import Field, ValidationError

from ai_support_agent.exceptions import ConfigurationError, EmbeddingRequestError, VectorStoreError
from ai_support_agent.rag.fallback_retriever import FallbackRetriever
from ai_support_agent.rag.retriever import Retriever
from ai_support_agent.rag.retriever import RetrievedChunk
from ai_support_agent.rag.runtime import (
    create_external_reference_gemini_vector_retriever,
    create_gemini_vector_retriever,
)
from ai_support_agent.service import (
    EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD,
    RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
    RETRIEVAL_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.order_status import (
    GetOrderStatusArguments,
    get_order_status,
)

SERVICE_OVERVIEW_URI = "support://service/overview"
SUPPORT_POLICY_ANSWER_PROMPT = "support-policy-answer"
SERVICE_OVERVIEW = """# AI Support Knowledge Base

Этот MCP-сервер предоставляет поиск по внутренним правилам поддержки.

## Доступный tool

- `search_knowledge_base(query)` — ищет релевантные фрагменты политик по вопросу и возвращает текст с источниками.

## Ограничения

- Сервер не предоставляет персональные данные, статусы заказов и операции изменения заказов.
- Результаты поиска содержат только найденные фрагменты правил; клиент должен показывать их как контекст, а не как неподтверждённый ответ от имени магазина.
"""
AUTHENTICATED_SERVICE_OVERVIEW = """# AI Support Knowledge Base

Этот MCP-сервер предоставляет поиск по внутренним правилам поддержки и чтение статуса заказа авторизованного пользователя.

## Доступные tools

- `search_knowledge_base(query)` — ищет релевантные фрагменты политик по вопросу и возвращает текст с источниками.
- `get_order_status(order_id)` — возвращает статус только заказа, доступного текущему авторизованному пользователю.

## Ограничения

- Идентификатор пользователя не принимается от модели как аргумент tool: он поступает в сервер только через доверенный контекст host.
- Для чужого и несуществующего заказа сервер возвращает одинаковый результат `order_not_found`.
- Сервер не выполняет операций изменения заказа.
"""


def create_support_mcp_server(retriever: Retriever) -> MCPServer:
    """Expose application retrieval through a standard MCP tool contract."""

    return _create_support_mcp_server(retriever, tool_context=None)


def create_authenticated_support_mcp_server(
    retriever: Retriever,
    tool_context: ToolExecutionContext,
) -> MCPServer:
    """Create a server that received user identity from a trusted host boundary."""

    return _create_support_mcp_server(retriever, tool_context=tool_context)


def _create_support_mcp_server(
    retriever: Retriever,
    *,
    tool_context: ToolExecutionContext | None,
) -> MCPServer:
    """Register only the tools allowed by the server's trusted context."""

    server = MCPServer(
        "AI Support Knowledge Base",
        instructions="Provides read-only support-policy fragments with source provenance.",
    )

    @server.resource(
        SERVICE_OVERVIEW_URI,
        name="service_overview",
        title="Обзор AI Support Knowledge Base",
        description="Краткое описание возможностей и границ этого MCP-сервера.",
        mime_type="text/markdown",
    )
    def get_service_overview() -> str:
        """Return a safe, static overview for a connected MCP client."""

        return (
            AUTHENTICATED_SERVICE_OVERVIEW
            if tool_context is not None
            else SERVICE_OVERVIEW
        )

    @server.prompt(
        name=SUPPORT_POLICY_ANSWER_PROMPT,
        title="Ответ по политике поддержки",
        description="Готовит сценарий поиска и ответа по правилам поддержки.",
    )
    def support_policy_answer(question: str) -> str:
        """Build a user-selected prompt; it does not call a tool or an LLM."""

        return (
            "Найди сведения по вопросу ниже с помощью tool "
            "search_knowledge_base. Затем дай краткий ответ только по найденным "
            "источникам. Если источников недостаточно, честно сообщи об этом.\n\n"
            f"Вопрос: {question}"
        )

    @server.tool(structured_output=True)
    def search_knowledge_base(query: str, debug: bool = False) -> dict[str, Any]:
        """Find relevant support-policy fragments for a user question."""

        chunks = retriever.retrieve(
            query,
            top_k=RETRIEVAL_TOP_K,
            threshold=RETRIEVAL_THRESHOLD,
            max_chunks_per_document=RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
        )
        result: dict[str, Any] = {
            "context": _build_display_context(chunks),
            "chunks": [
                {
                    "text": chunk.text,
                    "source_id": chunk.document_id,
                    "source_type": chunk.source_type.value,
                    "page_number": chunk.page_number,
                    "score": chunk.score,
                }
                for chunk in chunks
            ]
        }
        if debug:
            # Returned through the MCP response rather than stdout: stdio stdout
            # is reserved for protocol frames. This must stay opt-in because a
            # real user question can be sensitive.
            result["debug"] = {
                "received_query": query,
                "retrieved_chunk_count": len(chunks),
                "threshold": RETRIEVAL_THRESHOLD,
            }
        return result

    if tool_context is not None:

        @server.tool(
            name="get_order_status",
            description=(
                "Get status of one order visible to the authenticated user. "
                "Never accepts a user ID from the caller."
            ),
            structured_output=True,
        )
        def get_order_status_for_current_user(
            order_id: Annotated[
                str,
                Field(
                    min_length=6,
                    max_length=32,
                    pattern=r"^ORD-\d+$",
                    description="Public order number in the format ORD-12345.",
                ),
            ],
        ) -> dict[str, Any]:
            """Return order status only when the trusted user context owns it."""

            try:
                arguments = GetOrderStatusArguments(order_id=order_id)
            except ValidationError:
                return {
                    "ok": False,
                    "code": "invalid_arguments",
                    "message": "Order number has an invalid format.",
                    "retryable": False,
                }
            result = get_order_status(
                arguments,
                current_user_id=tool_context.current_user_id,
            )
            return result.model_dump(mode="json")

    return server


def _build_display_context(chunks: list[RetrievedChunk]) -> str:
    """Create a concise human-facing view while retaining raw chunks separately.

    Retrieval can return neighbouring chunks from one document.  They are useful
    as evidence, but showing each of them verbatim makes a small CLI response
    look repetitive.  The display view therefore uses the highest-ranked chunk
    for every source document; ``chunks`` still contains the complete evidence.
    """

    seen_sources: set[str] = set()
    excerpts: list[str] = []
    for chunk in chunks:
        if chunk.document_id in seen_sources:
            continue
        seen_sources.add(chunk.document_id)
        text = _strip_markdown_headings(chunk.text)
        if text:
            excerpts.append(text)
    return "\n\n".join(excerpts)


def _strip_markdown_headings(text: str) -> str:
    """Remove Markdown markers without discarding content on a joined line."""

    content_lines = []
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned.startswith("#"):
            # The chunker can join a heading and its first sentence onto one
            # line. Dropping that line would drop the actual evidence too.
            cleaned = cleaned.lstrip("#").strip()
        if cleaned:
            content_lines.append(cleaned)
    return " ".join(content_lines)


def create_support_retriever() -> FallbackRetriever:
    """Build the same retrieval policy used by the local support application."""

    return FallbackRetriever(
        primary=create_gemini_vector_retriever(),
        fallback_factory=create_external_reference_gemini_vector_retriever,
        fallback_threshold=EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD,
    )


def main() -> None:
    """Run the server over stdio; stdout remains reserved for MCP protocol frames."""

    try:
        create_support_mcp_server(create_support_retriever()).run(transport="stdio")
    except (ConfigurationError, EmbeddingRequestError, VectorStoreError) as error:
        raise SystemExit(f"Unable to start Support MCP server: {error}") from error


if __name__ == "__main__":
    main()
