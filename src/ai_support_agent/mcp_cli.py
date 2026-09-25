"""Manual command for exercising the Support MCP server through a real stdio client."""

import asyncio
import argparse

from ai_support_agent.config import load_gemini_embedding_config
from ai_support_agent.exceptions import ConfigurationError, EmbeddingRequestError
from ai_support_agent.mcp_client import SupportMcpClient
from ai_support_agent.mcp_server import SERVICE_OVERVIEW_URI, SUPPORT_POLICY_ANSWER_PROMPT


def main() -> None:
    """Run one read-only MCP discovery and search request."""

    parser = argparse.ArgumentParser(description="Search the local Support MCP server.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show the diagnostic payload returned by the MCP server.",
    )
    parser.add_argument(
        "--overview",
        action="store_true",
        help="Read and print the server overview resource without searching.",
    )
    parser.add_argument(
        "--policy-prompt",
        metavar="QUESTION",
        help="Render the support-policy-answer MCP prompt for a question.",
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            _run_search(
                SupportMcpClient(load_gemini_embedding_config()),
                debug=args.debug,
                show_overview=args.overview,
                policy_prompt_question=args.policy_prompt,
            )
        )
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
        return
    except EmbeddingRequestError:
        print("Не удалось получить ответ от MCP-сервера. Попробуйте позже.")
        return

async def _run_search(
    client: SupportMcpClient,
    *,
    debug: bool = False,
    show_overview: bool = False,
    policy_prompt_question: str | None = None,
) -> None:
    """Keep discovery and call_tool inside one live stdio session."""

    async with client.connect() as connection:
        listed_tools = await connection.list_tools()
        listed_resources = await connection.list_resources()
        listed_prompts = await connection.list_prompts()
        if show_overview:
            overview = await connection.read_resource(SERVICE_OVERVIEW_URI)
            for content in overview.contents:
                text = getattr(content, "text", None)
                if isinstance(text, str):
                    print(text)
            return
        if policy_prompt_question is not None:
            prompt = await connection.get_prompt(
                SUPPORT_POLICY_ANSWER_PROMPT,
                {"question": policy_prompt_question},
            )
            for message in prompt.messages:
                text = getattr(message.content, "text", None)
                if isinstance(text, str):
                    print(text)
            return
        tool_names = ", ".join(tool.name for tool in listed_tools.tools)
        print(f"Доступные MCP tools: {tool_names}")
        resource_uris = ", ".join(str(resource.uri) for resource in listed_resources.resources)
        print(f"Доступные MCP resources: {resource_uris}")
        prompt_names = ", ".join(prompt.name for prompt in listed_prompts.prompts)
        print(f"Доступные MCP prompts: {prompt_names}")
        query = await asyncio.to_thread(input, "Вопрос для MCP-поиска: ")
        query = query.strip()
        if not query:
            print("Введите вопрос и запустите программу снова.")
            return
        result = await client.search_in_connection(
            connection, query, listed_tools.tools, debug=debug
        )

    _print_result_without_discovery(result)
    if debug:
        print(f"[MCP debug] {result.payload.get('debug')}")


def _print_result_without_discovery(result: object) -> None:
    """Render concise context after discovery has already been printed."""

    payload = getattr(result, "payload")
    context = payload.get("context")
    if isinstance(context, str) and context:
        print(context)
        return

    # Compatibility fallback for an older MCP server that returns only chunks.
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        print("Релевантных фрагментов не найдено.")
        return
    print(_build_legacy_display_context(chunks))


def _build_legacy_display_context(chunks: list[object]) -> str:
    """Format old ``chunks``-only MCP responses without repeated raw excerpts."""

    seen_sources: set[object] = set()
    excerpts: list[str] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        source_id = chunk.get("source_id")
        if source_id in seen_sources:
            continue
        seen_sources.add(source_id)
        text = chunk.get("text")
        if not isinstance(text, str):
            continue
        content_lines = []
        for line in text.splitlines():
            cleaned = line.strip()
            if cleaned.startswith("#"):
                cleaned = cleaned.lstrip("#").strip()
            if cleaned:
                content_lines.append(cleaned)
        if content_lines:
            excerpts.append(" ".join(content_lines))
    return "\n\n".join(excerpts) or "Релевантных фрагментов не найдено."


if __name__ == "__main__":
    main()
