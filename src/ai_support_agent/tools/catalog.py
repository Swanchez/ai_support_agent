"""One application-owned catalog of executable tools and their exposure policies."""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from ai_support_agent.tools.context import ToolExecutionContext


class ToolEffect(StrEnum):
    """Whether a tool only reads state or can change it."""

    READ = "read"
    WRITE = "write"


class AgentToolAccess(StrEnum):
    """How a tool may be exposed to a bounded agent."""

    NONE = "none"
    READ = "read"
    PROPOSAL = "proposal"


ToolHandler = Callable[[BaseModel, ToolExecutionContext], BaseModel]
PROPOSAL_ONLY_DESCRIPTION = (
    "[proposal-only] This function does not execute the action. "
    "It only creates a proposal that requires explicit user confirmation."
)


@dataclass(frozen=True)
class RegisteredTool:
    """One validated bridge from an LLM-facing definition to application code."""

    definition: dict[str, object]
    arguments_model: type[BaseModel]
    handler: ToolHandler
    effect: ToolEffect = ToolEffect.READ
    agent_access: AgentToolAccess = AgentToolAccess.NONE


@dataclass(frozen=True)
class ToolCatalog:
    """Single source of truth for tool registrations and exposure metadata."""

    tools: Mapping[str, RegisteredTool]

    def definitions(self) -> list[dict[str, object]]:
        """Return all functions available to the regular confirmation-aware flow."""

        return [tool.definition for tool in self.tools.values()]

    def definitions_for_agent(self, access: AgentToolAccess) -> list[dict[str, object]]:
        """Return only tools granted to an agent for one explicit access level."""

        return [
            _agent_definition(tool.definition, access)
            for tool in self.tools.values()
            if tool.agent_access is access
        ]

    def names_for_agent(self, access: AgentToolAccess) -> frozenset[str]:
        """Return the application-owned names for one agent exposure policy."""

        return frozenset(
            name for name, tool in self.tools.items() if tool.agent_access is access
        )


def _agent_definition(
    definition: dict[str, object],
    access: AgentToolAccess,
) -> dict[str, object]:
    """Copy and annotate an LLM-facing definition without mutating the catalog."""

    agent_definition = deepcopy(definition)
    if access is AgentToolAccess.PROPOSAL:
        description = agent_definition.get("description", "")
        agent_definition["description"] = f"{PROPOSAL_ONLY_DESCRIPTION} {description}".strip()
    return agent_definition
