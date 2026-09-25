from ai_support_agent.mcp_client import McpSearchResult


def test_mcp_search_result_preserves_discovery_and_structured_payload() -> None:
    result = McpSearchResult(
        tool_names=("search_knowledge_base",),
        payload={"chunks": [{"source_id": "refund-policy-v1"}]},
    )

    assert result.tool_names == ("search_knowledge_base",)
    assert result.payload["chunks"] == [{"source_id": "refund-policy-v1"}]
