"""Safe audit records for application-owned tool execution."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol


class AuditEventType(StrEnum):
    """The application layer that produced an audit event."""

    TOOL_EXECUTION = "tool_execution"
    CONFIRMATION = "confirmation"


@dataclass(frozen=True)
class AuditEvent:
    """One security-relevant tool outcome without prompts or raw arguments."""

    occurred_at: datetime
    actor_id: str
    tool_name: str
    tool_effect: str | None
    event_type: AuditEventType
    outcome: str


class AuditSink(Protocol):
    """Storage boundary for audit events; a database implementation comes later."""

    def record(self, event: AuditEvent) -> None:
        """Persist or forward one safe event."""


@dataclass(frozen=True)
class NullAuditSink:
    """Default sink that keeps existing application behavior without persistence."""

    def record(self, event: AuditEvent) -> None:
        """Deliberately discard events until a host configures an audit backend."""

        _ = event


@dataclass
class InMemoryAuditSink:
    """Test and local-development sink; events disappear when the process ends."""

    events: list[AuditEvent] = field(default_factory=list)

    def record(self, event: AuditEvent) -> None:
        """Append one event in execution order."""

        self.events.append(event)


def create_audit_event(
    *,
    actor_id: str,
    tool_name: str,
    tool_effect: str | None,
    outcome: str,
    event_type: AuditEventType = AuditEventType.TOOL_EXECUTION,
    clock: Callable[[], datetime] | None = None,
) -> AuditEvent:
    """Create a UTC event while keeping a deterministic clock seam for tests."""

    occurred_at = (clock or (lambda: datetime.now(UTC)))()
    return AuditEvent(
        occurred_at=occurred_at,
        actor_id=actor_id,
        tool_name=tool_name,
        tool_effect=tool_effect,
        event_type=event_type,
        outcome=outcome,
    )
