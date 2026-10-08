"""Учебная среда: применяет политику до вызова исполнителя."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from harness.command_policy import Decision, PolicyResult, check_command
from harness.events import EventName, MemoryTrace, TraceEvent


class CommandExecutor(Protocol):
    def execute(self, argv: list[str], *, cwd: str) -> str:
        ...


class RecordingExecutor:
    """Только записывает вызовы. Не запускает процессы."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []
        self.directories: list[str] = []

    def execute(self, argv: list[str], *, cwd: str) -> str:
        self.commands.append(argv.copy())
        self.directories.append(cwd)
        return "Учебный исполнитель записал команду; процесс не запускался."


@dataclass(frozen=True)
class PendingCommand:
    request_id: str
    argv: tuple[str, ...]
    cwd: str
    reason: str


class ConfirmationNotFoundError(ValueError):
    """Запрос неизвестен или уже использован."""


@dataclass(frozen=True)
class RunResult:
    policy: PolicyResult
    output: str | None = None
    pending: PendingCommand | None = None
    operation_id: str = ""
    execution_limit_reached: bool = False


class CommandRuntime:
    def __init__(
        self, executor: CommandExecutor, *, trace: MemoryTrace | None = None,
        max_executions: int = 3,
    ) -> None:
        if type(max_executions) is not int or max_executions < 0:
            raise ValueError("max_executions должен быть целым неотрицательным числом.")
        self._executor = executor
        self._max_executions = max_executions
        self._used_executions = 0
        self.trace = trace if trace is not None else MemoryTrace()
        self._pending: dict[str, PendingCommand] = {}
        self._lock = Lock()

    @property
    def max_executions(self) -> int:
        return self._max_executions

    @property
    def used_executions(self) -> int:
        with self._lock:
            return self._used_executions

    def run(self, argv: list[str], *, cwd: str = ".") -> RunResult:
        operation_id = uuid4().hex
        self.trace.record(TraceEvent(operation_id, EventName.COMMAND_RECEIVED))
        # Проверяем и передаём исполнителю одну и ту же копию аргументов.
        # Некорректный вход передаём валидатору без попытки копирования.
        command = argv.copy() if isinstance(argv, list) else argv
        policy = check_command(command)
        if policy.decision == Decision.DENY:
            self.trace.record(TraceEvent(operation_id, EventName.POLICY_DENIED, Decision.DENY))
            return RunResult(policy=policy, operation_id=operation_id)

        directory = str(Path(cwd).resolve())
        if policy.decision == Decision.ASK:
            pending = PendingCommand(operation_id, tuple(command), directory, policy.reason)
            with self._lock:
                self._pending[pending.request_id] = pending
            self.trace.record(TraceEvent(operation_id, EventName.CONFIRMATION_REQUESTED, Decision.ASK))
            return RunResult(policy=policy, pending=pending, operation_id=operation_id)

        self.trace.record(TraceEvent(operation_id, EventName.POLICY_ALLOWED, Decision.ALLOW))
        output = self._execute(command, directory, operation_id)
        return RunResult(
            policy=policy, output=output, operation_id=operation_id,
            execution_limit_reached=output is None,
        )

    def _execute(self, command: list[str], directory: str, operation_id: str) -> str | None:
        # Проверка и резервирование атомарны. Сам исполнитель работает вне lock.
        # Слот расходуется и при ошибке; автоматического возврата бюджета нет.
        with self._lock:
            exhausted = self._used_executions >= self._max_executions
            if not exhausted:
                self._used_executions += 1
        if exhausted:
            self.trace.record(TraceEvent(operation_id, EventName.EXECUTION_LIMIT_REACHED))
            return None
        self.trace.record(TraceEvent(operation_id, EventName.EXECUTION_STARTED))
        started = perf_counter()
        try:
            output = self._executor.execute(command, cwd=directory)
        except Exception as error:
            self.trace.record(TraceEvent(
                operation_id, EventName.EXECUTION_FAILED,
                duration_ms=(perf_counter() - started) * 1000,
                error_type=type(error).__name__,
            ))
            raise
        self.trace.record(TraceEvent(
            operation_id, EventName.EXECUTION_COMPLETED,
            duration_ms=(perf_counter() - started) * 1000,
        ))
        return output

    def _consume(self, request_id: str) -> PendingCommand:
        # Удаляем до execute: даже при ошибке исполнителя токен не переиспользуется.
        with self._lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            raise ConfirmationNotFoundError("Запрос неизвестен или уже использован.")
        return pending

    def confirm(self, request_id: str) -> RunResult:
        """Вызывается доверенным интерфейсом пользователя, не самой моделью."""
        pending = self._consume(request_id)
        self.trace.record(TraceEvent(request_id, EventName.CONFIRMATION_ACCEPTED))
        command = list(pending.argv)
        # Новое запрещающее правило нельзя обойти старым подтверждением.
        policy = check_command(command)
        if policy.decision == Decision.DENY:
            self.trace.record(TraceEvent(request_id, EventName.POLICY_DENIED, Decision.DENY))
            return RunResult(policy=policy, operation_id=request_id)
        output = self._execute(command, pending.cwd, request_id)
        return RunResult(
            policy=PolicyResult(Decision.ALLOW, "Сохранённая операция подтверждена."),
            output=output,
            operation_id=request_id,
            execution_limit_reached=output is None,
        )

    def reject(self, request_id: str) -> PendingCommand:
        """Отклонить запрос без выполнения команды."""
        pending = self._consume(request_id)
        self.trace.record(TraceEvent(request_id, EventName.CONFIRMATION_REJECTED))
        return pending
