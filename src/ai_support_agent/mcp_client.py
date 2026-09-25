"""Thin MCP client for launching the local Support MCP server over stdio."""

import asyncio
import sys
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass

from mcp import Client, StdioServerParameters

from ai_support_agent.config import GeminiEmbeddingConfig, PROJECT_ROOT
from ai_support_agent.exceptions import EmbeddingRequestError


@dataclass(frozen=True)
class McpSearchResult:
    """Only the discovery result and structured tool payload needed by this client."""

    tool_names: tuple[str, ...]
    payload: dict[str, object]


@dataclass(frozen=True)
class SupportMcpClient:
    """Launch the local MCP server as a child process with only required config."""

    embedding_config: GeminiEmbeddingConfig

    def search_knowledge_base(self, query: str) -> McpSearchResult:
        """Discover server tools and invoke the read-only RAG tool over stdio."""

        return asyncio.run(self._search_knowledge_base(query))

    async def _search_knowledge_base(self, query: str) -> McpSearchResult:
        """Own the MCP connection lifecycle; the child process ends with this block."""

        async with self.connect() as client:
            listed_tools = await client.list_tools()
            return await self.search_in_connection(client, query, listed_tools.tools)

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Client]:
        """Open one stdio MCP connection and clean up the child process on exit."""

        async with Client(self._server_parameters()) as client:
            yield client

    async def search_in_connection(
        self,
        client: Client,
        query: str,
        tools: list[object],
        *,
        debug: bool = False,
    ) -> McpSearchResult:
        """Call the RAG tool after the caller has already completed discovery."""

        result = await client.call_tool(
            "search_knowledge_base", {"query": query, "debug": debug}
        )
        if result.is_error or not isinstance(result.structured_content, dict):
            raise EmbeddingRequestError("Support MCP search tool failed.")
        return McpSearchResult(
            tool_names=tuple(getattr(tool, "name") for tool in tools),
            payload=result.structured_content,
        )

    def _server_parameters(self) -> StdioServerParameters:
        """Pass only the embedding credentials needed by the local server process."""

        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "ai_support_agent.mcp_server"],
            env={
                "GEMINI_API_KEY": self.embedding_config.api_key,
                "GEMINI_EMBEDDING_MODEL": self.embedding_config.model,
            },
            cwd=PROJECT_ROOT,
        )
