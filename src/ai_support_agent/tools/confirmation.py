"""Application-owned, one-time confirmations for write-capable tool actions."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from ai_support_agent.tools.audit import (
    AuditEventType,
    AuditSink,
    NullAuditSink,
    create_audit_event,
)


@dataclass(frozen=True)
class PendingToolAction:
    """A validated action waiting for its owning user to approve it."""

    confirmation_id: str
    user_id: str
    tool_name: str
    arguments: dict[str, Any]
    conversation_id: str | None = None


@dataclass
class InMemoryPendingActionStore:
    """Learning-only store; production requires durable storage and expiration."""

    _actions: dict[str, PendingToolAction] = field(default_factory=dict)
    audit_sink: AuditSink = field(default_factory=NullAuditSink)

    def create(
        self,
        *,
        user_id: str,
        conversation_id: str | None = None,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PendingToolAction:
        """Save a new action and return an unpredictable application-generated ID."""

        for pending_id, pending in tuple(self._actions.items()):
            if pending.user_id == user_id and pending.conversation_id == conversation_id:
                del self._actions[pending_id]

        action = PendingToolAction(
            confirmation_id=uuid4().hex,
            user_id=user_id,
            tool_name=tool_name,
            arguments=deepcopy(arguments),
            conversation_id=conversation_id,
        )
        self._actions[action.confirmation_id] = action
        self._record(action, "proposal_created")
        return action

    def find_for_user(
        self, user_id: str, conversation_id: str | None = None
    ) -> PendingToolAction | None:
        """Return one pending action in this user's current conversation scope."""

        return next(
            (
                action
                for action in self._actions.values()
                if action.user_id == user_id and action.conversation_id == conversation_id
            ),
            None,
        )

    def discard_for_user(
        self,
        *,
        confirmation_id: str,
        user_id: str,
        conversation_id: str | None = None,
    ) -> bool:
        """Discard one pending action only when it belongs to the requesting user."""

        action = self._actions.get(confirmation_id)
        if (
            action is None
            or action.user_id != user_id
            or action.conversation_id != conversation_id
        ):
            return False
        del self._actions[confirmation_id]
        self._record(action, "confirmation_rejected")
        return True

    def approve_for_user(
        self,
        *,
        confirmation_id: str,
        user_id: str,
        conversation_id: str | None = None,
    ) -> PendingToolAction | None:
        """Consume and return an action only when it belongs to the approving user."""

        action = self._actions.get(confirmation_id)
        if (
            action is None
            or action.user_id != user_id
            or action.conversation_id != conversation_id
        ):
            return None
        approved = self._actions.pop(confirmation_id)
        self._record(approved, "confirmation_approved")
        return approved

    def record_unclear_confirmation(self, action: PendingToolAction) -> None:
        """Record a non-destructive reply that leaves a pending action in place."""

        self._record(action, "confirmation_unclear")

    def _record(self, action: PendingToolAction, outcome: str) -> None:
        """Write confirmation lifecycle metadata without storing action arguments."""

        self.audit_sink.record(
            create_audit_event(
                actor_id=action.user_id,
                tool_name=action.tool_name,
                tool_effect="write",
                event_type=AuditEventType.CONFIRMATION,
                outcome=outcome,
            )
        )
