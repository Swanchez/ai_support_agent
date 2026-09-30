from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest
from sqlalchemy.exc import OperationalError

from ai_support_agent.persistence.audit_sink import PostgresAuditSink
from ai_support_agent.persistence.models import AuditEventRecord
from ai_support_agent.tools.audit import AuditEvent, AuditEventType, AuditSinkUnavailable


@dataclass
class FakeSession:
    records: list[AuditEventRecord] = field(default_factory=list)
    committed: bool = False

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args

    def add(self, record: AuditEventRecord) -> None:
        self.records.append(record)

    def commit(self) -> None:
        self.committed = True


def test_postgres_audit_sink_persists_only_the_safe_audit_contract() -> None:
    session = FakeSession()
    sink = PostgresAuditSink(lambda: session)  # type: ignore[arg-type]
    event = AuditEvent(
        occurred_at=datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
        actor_id="demo-user-1",
        tool_name="cancel_order",
        tool_effect="write",
        event_type=AuditEventType.CONFIRMATION,
        outcome="confirmation_approved",
    )

    sink.record(event)

    assert session.committed is True
    assert len(session.records) == 1
    record = session.records[0]
    assert record.actor_id == "demo-user-1"
    assert record.event_type == "confirmation"
    assert record.outcome == "confirmation_approved"
    assert not hasattr(record, "arguments")
    assert not hasattr(record, "prompt")


def test_postgres_audit_sink_exposes_storage_failure_to_the_composition_layer() -> None:
    @dataclass
    class UnavailableSession(FakeSession):
        def commit(self) -> None:
            raise OperationalError("INSERT", {}, RuntimeError("connection lost"))

    sink = PostgresAuditSink(lambda: UnavailableSession())  # type: ignore[arg-type]
    event = AuditEvent(
        occurred_at=datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
        actor_id="demo-user-1",
        tool_name="get_order_status",
        tool_effect="read",
        event_type=AuditEventType.TOOL_EXECUTION,
        outcome="success",
    )

    with pytest.raises(AuditSinkUnavailable):
        sink.record(event)
