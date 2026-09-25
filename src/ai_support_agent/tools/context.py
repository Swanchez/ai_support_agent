"""Trusted request context that must never be supplied by an LLM."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolExecutionContext:
    """Identity established by the application after authenticating the user."""

    current_user_id: str
    idempotency_key: str | None = None


DEMO_TOOL_CONTEXT = ToolExecutionContext(current_user_id="demo-user-1")
