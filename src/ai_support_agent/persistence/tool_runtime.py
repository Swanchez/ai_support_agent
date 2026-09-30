"""Composition root for tools backed by the local PostgreSQL database."""

from collections.abc import Callable

from sqlalchemy.orm import Session

from ai_support_agent.persistence.audit_sink import PostgresAuditSink
from ai_support_agent.persistence.order_repository import PostgresOrderRepository
from ai_support_agent.tools.audit import BestEffortAuditSink
from ai_support_agent.tools.default_catalog import create_tool_catalog
from ai_support_agent.tools.executor import ToolExecutor


def create_persistent_tool_executor(
    session_factory: Callable[[], Session],
) -> ToolExecutor:
    """Compose production-style order tools without changing their core contracts."""

    order_repository = PostgresOrderRepository(session_factory)
    catalog = create_tool_catalog(order_repository)
    audit_sink = BestEffortAuditSink(PostgresAuditSink(session_factory))
    return ToolExecutor(registry=catalog.tools, audit_sink=audit_sink)
