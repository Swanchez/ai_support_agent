"""PostgreSQL adapter for the existing safe audit-event contract."""

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ai_support_agent.persistence.models import AuditEventRecord
from ai_support_agent.tools.audit import AuditEvent, AuditSinkUnavailable


@dataclass(frozen=True)
class PostgresAuditSink:
    """Store already-sanitized audit events without prompts, arguments, or secrets."""

    session_factory: Callable[[], Session]

    def record(self, event: AuditEvent) -> None:
        """Persist one event in a short independent database transaction."""

        record = AuditEventRecord(
            occurred_at=event.occurred_at,
            actor_id=event.actor_id,
            tool_name=event.tool_name,
            tool_effect=event.tool_effect,
            event_type=event.event_type.value,
            outcome=event.outcome,
        )
        try:
            with self.session_factory() as session:
                session.add(record)
                session.commit()
        except SQLAlchemyError as error:
            raise AuditSinkUnavailable("PostgreSQL audit storage is unavailable.") from error
