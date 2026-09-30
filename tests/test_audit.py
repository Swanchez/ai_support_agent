import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from ai_support_agent.tools.audit import (
    AuditEvent,
    AuditEventType,
    AuditSinkUnavailable,
    BestEffortAuditSink,
)


@dataclass(frozen=True)
class UnavailableAuditSink:
    def record(self, event: AuditEvent) -> None:
        _ = event
        raise AuditSinkUnavailable("database is down")


def test_best_effort_audit_does_not_hide_a_completed_application_result(
    caplog: object,
) -> None:
    event = AuditEvent(
        occurred_at=datetime(2026, 9, 29, 12, 0, tzinfo=UTC),
        actor_id="demo-user-1",
        tool_name="cancel_order",
        tool_effect="write",
        event_type=AuditEventType.TOOL_EXECUTION,
        outcome="success",
    )

    with caplog.at_level(logging.WARNING):  # type: ignore[union-attr]
        BestEffortAuditSink(UnavailableAuditSink()).record(event)

    assert "Audit event was not persisted" in caplog.text  # type: ignore[union-attr]
