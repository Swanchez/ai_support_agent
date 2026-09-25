"""Adapters that expose existing read capabilities as safe agent observations."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ai_support_agent.agents.core import AgentToolResult
from ai_support_agent.rag.retriever import Retriever
from ai_support_agent.service import (
    RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
    RETRIEVAL_THRESHOLD,
    RETRIEVAL_TOP_K,
)
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.executor import ToolExecutor
from ai_support_agent.tools.catalog import AgentToolAccess, ToolCatalog
from ai_support_agent.tools.default_catalog import DEFAULT_TOOL_CATALOG


class SearchKnowledgeBaseArguments(BaseModel):
    """The only agent-provided input accepted by the RAG search action."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)


@dataclass
class SearchKnowledgeBaseTool:
    """Expose bounded semantic retrieval as one read-only agent action."""

    retriever: Retriever

    def execute(self, arguments: dict[str, Any]) -> AgentToolResult:
        """Validate a query, retrieve bounded evidence, and preserve provenance."""

        try:
            validated = SearchKnowledgeBaseArguments.model_validate(arguments)
        except ValidationError:
            return AgentToolResult({"ok": False, "code": "invalid_tool_arguments"})

        chunks = self.retriever.retrieve(
            validated.query,
            top_k=RETRIEVAL_TOP_K,
            threshold=RETRIEVAL_THRESHOLD,
            max_chunks_per_document=RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
        )
        return AgentToolResult(
            data={
                "ok": True,
                "chunks": [
                    {
                        "text": chunk.text,
                        "source_id": chunk.document_id,
                        "source_type": chunk.source_type.value,
                        "page_number": chunk.page_number,
                    }
                    for chunk in chunks
                ],
            },
            source_ids=tuple(dict.fromkeys(chunk.document_id for chunk in chunks)),
        )


@dataclass
class OrderStatusAgentTool:
    """Adapt the existing application executor to the agent observation contract."""

    executor: ToolExecutor
    context: ToolExecutionContext

    def execute(self, arguments: dict[str, Any]) -> AgentToolResult:
        """Run only the existing read-only order status action through its safeguards."""

        result = self.executor.execute("get_order_status", arguments, self.context)
        return AgentToolResult(data=result, source_ids=("get_order_status",))


def search_knowledge_base_tool_definition() -> dict[str, object]:
    """Describe the bounded RAG search action for a provider-specific agent planner."""

    return {
        "type": "function",
        "name": "search_knowledge_base",
        "description": "Searches the support knowledge base for relevant policy fragments.",
        "parameters": SearchKnowledgeBaseArguments.model_json_schema(),
    }


def read_agent_tool_definitions(
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> list[dict[str, object]]:
    """Return only agent-safe read actions; write actions stay behind confirm-flow."""

    return [
        search_knowledge_base_tool_definition(),
        *catalog.definitions_for_agent(AgentToolAccess.READ),
    ]


def agent_tool_definitions(
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> list[dict[str, object]]:
    """Expose read tools plus proposal-only writes; execution stays read-only."""

    return [
        *read_agent_tool_definitions(catalog),
        *catalog.definitions_for_agent(AgentToolAccess.PROPOSAL),
    ]


def proposal_agent_tool_names(
    catalog: ToolCatalog = DEFAULT_TOOL_CATALOG,
) -> frozenset[str]:
    """Return names that planner calls must become proposals, never executions."""

    return catalog.names_for_agent(AgentToolAccess.PROPOSAL)
