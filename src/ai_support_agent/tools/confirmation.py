"""Application-owned, one-time confirmations for write-capable tool actions."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class PendingToolAction:
    """A validated action waiting for its owning user to approve it."""

    confirmation_id: str
    user_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class InMemoryPendingActionStore:
    """Learning-only store; production requires durable storage and expiration."""

    _actions: dict[str, PendingToolAction] = field(default_factory=dict)

    def create(
        self,
        *,
        user_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> PendingToolAction:
        """Save a new action and return an unpredictable application-generated ID."""

        for pending_id, pending in tuple(self._actions.items()):
            if pending.user_id == user_id:
                del self._actions[pending_id]

        action = PendingToolAction(
            confirmation_id=uuid4().hex,
            user_id=user_id,
            tool_name=tool_name,
            arguments=deepcopy(arguments),
        )
        self._actions[action.confirmation_id] = action
        return action

    def find_for_user(self, user_id: str) -> PendingToolAction | None:
        """Return the user's sole pending action without consuming it."""

        return next((action for action in self._actions.values() if action.user_id == user_id), None)

    def discard_for_user(self, *, confirmation_id: str, user_id: str) -> bool:
        """Discard one pending action only when it belongs to the requesting user."""

        action = self._actions.get(confirmation_id)
        if action is None or action.user_id != user_id:
            return False
        del self._actions[confirmation_id]
        return True

    def approve_for_user(
        self,
        *,
        confirmation_id: str,
        user_id: str,
    ) -> PendingToolAction | None:
        """Consume and return an action only when it belongs to the approving user."""

        action = self._actions.get(confirmation_id)
        if action is None or action.user_id != user_id:
            return None
        return self._actions.pop(confirmation_id)
