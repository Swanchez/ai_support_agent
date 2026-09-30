from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from ai_support_agent.persistence.models import AuditEventRecord, OrderRecord
from ai_support_agent.persistence.tool_runtime import create_persistent_tool_executor
from ai_support_agent.tools.context import ToolExecutionContext


@dataclass
class ReadSession:
    order: OrderRecord

    def __enter__(self) -> "ReadSession":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args

    def scalar(self, statement: object) -> OrderRecord:
        _ = statement
        return self.order


@dataclass
class AuditSession:
    records: list[AuditEventRecord] = field(default_factory=list)
    committed: bool = False

    def __enter__(self) -> "AuditSession":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args

    def add(self, record: AuditEventRecord) -> None:
        self.records.append(record)

    def commit(self) -> None:
        self.committed = True


def test_persistent_tool_composition_reads_from_postgres_and_records_audit() -> None:
    read_session = ReadSession(
        OrderRecord(
            id="ORD-1001",
            user_id="demo-user-1",
            status="shipped",
            estimated_delivery_at=date(2026, 9, 24),
            updated_at=datetime(2026, 9, 20, tzinfo=UTC),
        )
    )
    audit_session = AuditSession()
    sessions = iter((read_session, audit_session))
    executor = create_persistent_tool_executor(lambda: next(sessions))  # type: ignore[arg-type]

    result = executor.execute(
        "get_order_status",
        {"order_id": "ORD-1001"},
        ToolExecutionContext(current_user_id="demo-user-1"),
    )

    assert result["ok"] is True
    assert audit_session.committed is True
    assert len(audit_session.records) == 1
    assert audit_session.records[0].tool_name == "get_order_status"
    assert audit_session.records[0].outcome == "success"
