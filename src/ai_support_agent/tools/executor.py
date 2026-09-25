"""Trusted execution boundary for model-requested application tools."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ai_support_agent.tools.catalog import (
    RegisteredTool,
    ToolCatalog,
    ToolEffect,
)
from ai_support_agent.tools.audit import AuditSink, NullAuditSink, create_audit_event
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.default_catalog import DEFAULT_TOOL_CATALOG


class ExecutorErrorCode(StrEnum):
    """Failures caused before a registered tool can run."""

    TOOL_NOT_ALLOWED = "tool_not_allowed"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    CONFIRMATION_REQUIRED = "confirmation_required"
    IDEMPOTENCY_KEY_REQUIRED = "idempotency_key_required"


class ExecutorFailure(BaseModel):
    """A safe error payload for an unknown tool or invalid model arguments."""

    model_config = ConfigDict(extra="forbid")

    ok: Literal[False] = False
    code: ExecutorErrorCode
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class ArgumentValidation:
    """Validated tool arguments or a safe pre-execution failure."""

    arguments: BaseModel | None = None
    failure: ExecutorFailure | None = None

    def __post_init__(self) -> None:
        if (self.arguments is None) == (self.failure is None):
            raise ValueError("Argument validation must contain arguments or one failure.")


@dataclass(frozen=True)
class ToolExecutor:
    """Validates requests and invokes only tools present in its explicit registry."""

    registry: Mapping[str, RegisteredTool]
    audit_sink: AuditSink = field(default_factory=NullAuditSink)

    def definitions(self) -> list[dict[str, object]]:
        """Return the public tool descriptions that may be sent to an LLM provider."""

        return [tool.definition for tool in self.registry.values()]

    def requires_confirmation(self, tool_name: str) -> bool:
        """Return whether this registered tool can change application state."""

        tool = self.registry.get(tool_name)
        return tool is not None and tool.effect is ToolEffect.WRITE

    def is_registered(self, tool_name: str) -> bool:
        """Return whether a tool name is present in the explicit application whitelist."""

        return tool_name in self.registry

    def validate_arguments(self, tool_name: str, raw_arguments: object) -> ArgumentValidation:
        """Validate untrusted arguments before creating a pending write action."""

        tool = self.registry.get(tool_name)
        if tool is None:
            return ArgumentValidation(
                failure=ExecutorFailure(
                    code=ExecutorErrorCode.TOOL_NOT_ALLOWED,
                    message="Requested tool is not available.",
                )
            )
        try:
            return ArgumentValidation(arguments=tool.arguments_model.model_validate(raw_arguments))
        except ValidationError:
            return ArgumentValidation(
                failure=ExecutorFailure(
                    code=ExecutorErrorCode.INVALID_TOOL_ARGUMENTS,
                    message="Tool arguments do not match the declared schema.",
                )
            )

    def execute(
        self,
        tool_name: str,
        raw_arguments: object,
        context: ToolExecutionContext,
        *,
        confirmation_granted: bool = False,
    ) -> dict[str, Any]:
        """Run one allowed tool and serialize its structured result for the model."""

        validation = self.validate_arguments(tool_name, raw_arguments)
        if validation.failure is not None:
            result = validation.failure.model_dump(mode="json")
            self._record_outcome(tool_name, context, None, result)
            return result

        tool = self.registry[tool_name]

        if tool.effect is ToolEffect.WRITE and not confirmation_granted:
            result = ExecutorFailure(
                code=ExecutorErrorCode.CONFIRMATION_REQUIRED,
                message="This action requires explicit user confirmation.",
            ).model_dump(mode="json")
            self._record_outcome(tool_name, context, tool.effect, result)
            return result

        if tool.effect is ToolEffect.WRITE and not context.idempotency_key:
            result = ExecutorFailure(
                code=ExecutorErrorCode.IDEMPOTENCY_KEY_REQUIRED,
                message="Write actions require an application-generated idempotency key.",
            ).model_dump(mode="json")
            self._record_outcome(tool_name, context, tool.effect, result)
            return result

        assert validation.arguments is not None
        result = tool.handler(validation.arguments, context).model_dump(mode="json")
        self._record_outcome(tool_name, context, tool.effect, result)
        return result

    def _record_outcome(
        self,
        tool_name: str,
        context: ToolExecutionContext,
        effect: ToolEffect | None,
        result: Mapping[str, Any],
    ) -> None:
        """Record one safe outcome after every execution attempt."""

        outcome = "success" if result.get("ok") is True else str(result["code"])
        self.audit_sink.record(
            create_audit_event(
                actor_id=context.current_user_id,
                tool_name=tool_name,
                tool_effect=effect.value if effect is not None else None,
                outcome=outcome,
            )
        )


DEFAULT_TOOL_EXECUTOR = ToolExecutor(
    registry=DEFAULT_TOOL_CATALOG.tools
)
