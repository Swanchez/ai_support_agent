from ai_support_agent.tools.catalog import AgentToolAccess
from ai_support_agent.tools.catalog import PROPOSAL_ONLY_DESCRIPTION
from ai_support_agent.tools.context import DEMO_TOOL_CONTEXT
from ai_support_agent.tools.default_catalog import (
    DEFAULT_TOOL_CATALOG,
    create_tool_catalog,
)
from ai_support_agent.tools.executor import DEFAULT_TOOL_EXECUTOR, ToolExecutor
from ai_support_agent.tools.order_status import (
    InMemoryOrderRepository,
    OrderStatus,
    OrderStatusData,
    StoredOrder,
)


def test_catalog_is_the_single_registration_source_for_default_executor() -> None:
    assert DEFAULT_TOOL_EXECUTOR.registry is DEFAULT_TOOL_CATALOG.tools
    assert [definition["name"] for definition in DEFAULT_TOOL_CATALOG.definitions()] == [
        "get_order_status",
        "cancel_order",
    ]


def test_catalog_exposes_only_read_tools_to_the_agent() -> None:
    assert [
        definition["name"]
        for definition in DEFAULT_TOOL_CATALOG.definitions_for_agent(AgentToolAccess.READ)
    ] == ["get_order_status"]
    assert [
        definition["name"]
        for definition in DEFAULT_TOOL_CATALOG.definitions_for_agent(AgentToolAccess.PROPOSAL)
    ] == ["cancel_order"]
    proposal_definition = DEFAULT_TOOL_CATALOG.definitions_for_agent(
        AgentToolAccess.PROPOSAL
    )[0]
    assert proposal_definition["description"].startswith(PROPOSAL_ONLY_DESCRIPTION)
    assert "[proposal-only]" not in DEFAULT_TOOL_CATALOG.tools["cancel_order"].definition[
        "description"
    ]


def test_catalog_uses_the_order_repository_given_to_its_factory() -> None:
    repository = InMemoryOrderRepository(
        {
            "ORD-2001": StoredOrder(
                owner_user_id="demo-user-1",
                data=OrderStatusData(
                    order_id="ORD-2001",
                    status=OrderStatus.DELIVERED,
                    updated_at="2026-09-25",
                ),
            )
        }
    )
    executor = ToolExecutor(registry=create_tool_catalog(repository).tools)

    result = executor.execute(
        "get_order_status",
        {"order_id": "ORD-2001"},
        DEMO_TOOL_CONTEXT,
    )

    assert result["ok"] is True
    assert result["order"]["status"] == "delivered"
