"""Структурированные события без аргументов, путей и текстов исключений."""

from dataclasses import dataclass
from enum import StrEnum
from threading import Lock

from harness.command_policy import Decision


class EventName(StrEnum):
    COMMAND_RECEIVED = "command_received"
    POLICY_ALLOWED = "policy_allowed"
    POLICY_DENIED = "policy_denied"
    CONFIRMATION_REQUESTED = "confirmation_requested"
    CONFIRMATION_ACCEPTED = "confirmation_accepted"
    CONFIRMATION_REJECTED = "confirmation_rejected"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    EXECUTION_LIMIT_REACHED = "execution_limit_reached"


@dataclass(frozen=True)
class TraceEvent:
    operation_id: str
    event: EventName
    decision: Decision | None = None
    duration_ms: float | None = None
    error_type: str | None = None


class MemoryTrace:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []
        self._lock = Lock()

    def record(self, event: TraceEvent) -> None:
        with self._lock:
            self._events.append(event)

    def for_operation(self, operation_id: str) -> tuple[TraceEvent, ...]:
        with self._lock:
            return tuple(event for event in self._events if event.operation_id == operation_id)
