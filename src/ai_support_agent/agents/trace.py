"""Safe, compact developer trace for a bounded agent run."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ai_support_agent.agents.core import AgentState


_SENSITIVE_ARGUMENT_MARKERS = ("api_key", "secret", "token", "password")


@dataclass(frozen=True)
class AgentTraceEntry:
    """One executed action without raw prompts, reasoning, or document contents."""

    step: int
    action_name: str
    arguments: dict[str, Any]
    ok: bool | None
    code: str | None
    source_ids: tuple[str, ...]
    chunk_count: int | None


def build_agent_trace(state: AgentState) -> list[AgentTraceEntry]:
    """Convert application-owned observations into safe diagnostic entries."""

    entries: list[AgentTraceEntry] = []
    for step, observation in enumerate(state.observations, start=1):
        data = observation.result.data
        ok = data.get("ok") if isinstance(data.get("ok"), bool) else None
        code = data.get("code") if isinstance(data.get("code"), str) else None
        chunks = data.get("chunks")
        entries.append(
            AgentTraceEntry(
                step=step,
                action_name=observation.action.name,
                arguments=_safe_arguments(observation.action.arguments),
                ok=ok,
                code=code,
                source_ids=observation.result.source_ids,
                chunk_count=len(chunks) if isinstance(chunks, list) else None,
            )
        )
    return entries


def format_agent_trace(state: AgentState) -> str:
    """Format only a concise developer trace; never expose model reasoning."""

    lines = ["[Agent trace]"]
    for entry in build_agent_trace(state):
        outcome = "ok" if entry.ok is True else "error" if entry.ok is False else "unknown"
        details = [outcome]
        if entry.code:
            details.append(f"code={entry.code}")
        if entry.chunk_count is not None:
            details.append(f"chunks={entry.chunk_count}")
        if entry.source_ids:
            details.append(f"sources={', '.join(entry.source_ids)}")
        lines.append(
            f"{entry.step}. {entry.action_name}({entry.arguments}) -> "
            f"{'; '.join(details)}"
        )
    return "\n".join(lines)


def _safe_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Redact common secret fields before a debug value reaches the console."""

    return {
        key: "[redacted]" if _is_sensitive(key) else value
        for key, value in arguments.items()
    }


def _is_sensitive(key: str) -> bool:
    """Recognise conventional secret-bearing argument names."""

    lowered = key.lower()
    return any(marker in lowered for marker in _SENSITIVE_ARGUMENT_MARKERS)
