from dataclasses import asdict

import pytest

from harness.events import EventName as E
from harness.runtime import CommandRuntime, RecordingExecutor


def names(runtime, operation_id):
    return [event.event for event in runtime.trace.for_operation(operation_id)]


def test_allowed_trace_and_duration():
    runtime = CommandRuntime(RecordingExecutor())
    result = runtime.run(["docker", "compose", "ps"])
    assert names(runtime, result.operation_id) == [
        E.COMMAND_RECEIVED, E.POLICY_ALLOWED, E.EXECUTION_STARTED, E.EXECUTION_COMPLETED,
    ]
    assert runtime.trace.for_operation(result.operation_id)[-1].duration_ms >= 0


def test_denied_trace_has_no_execution():
    runtime = CommandRuntime(RecordingExecutor())
    result = runtime.run(["pwsh"])
    assert names(runtime, result.operation_id) == [E.COMMAND_RECEIVED, E.POLICY_DENIED]


def test_waiting_and_confirmed_share_operation_id():
    runtime = CommandRuntime(RecordingExecutor())
    result = runtime.run(["unknown"])
    assert names(runtime, result.operation_id) == [E.COMMAND_RECEIVED, E.CONFIRMATION_REQUESTED]
    confirmed = runtime.confirm(result.pending.request_id)
    assert confirmed.operation_id == result.operation_id
    assert names(runtime, result.operation_id)[2:] == [
        E.CONFIRMATION_ACCEPTED, E.EXECUTION_STARTED, E.EXECUTION_COMPLETED,
    ]


def test_rejection_is_not_execution_failure():
    runtime = CommandRuntime(RecordingExecutor())
    result = runtime.run(["unknown"])
    runtime.reject(result.pending.request_id)
    assert names(runtime, result.operation_id) == [
        E.COMMAND_RECEIVED, E.CONFIRMATION_REQUESTED, E.CONFIRMATION_REJECTED,
    ]


def test_error_type_logged_without_error_text_or_inputs():
    class FailingExecutor:
        def execute(self, argv, *, cwd):
            raise ValueError("secret-error-text")

    runtime = CommandRuntime(FailingExecutor())
    result = runtime.run(["secret-command-argument"], cwd="secret-directory")
    with pytest.raises(ValueError):
        runtime.confirm(result.pending.request_id)
    events = runtime.trace.for_operation(result.operation_id)
    assert events[-1].event == E.EXECUTION_FAILED
    assert events[-1].error_type == "ValueError"
    assert events[-1].duration_ms >= 0
    serialized = repr([asdict(event) for event in events])
    assert "secret" not in serialized
    assert E.EXECUTION_COMPLETED not in names(runtime, result.operation_id)


def test_trace_snapshot_and_operations_are_separate():
    runtime = CommandRuntime(RecordingExecutor())
    first = runtime.run(["unknown"])
    snapshot = runtime.trace.for_operation(first.operation_id)
    second = runtime.run(["pwsh"])
    runtime.reject(first.pending.request_id)
    assert len(snapshot) == 2
    assert all(event.operation_id == second.operation_id for event in runtime.trace.for_operation(second.operation_id))


def test_cli_shows_trace():
    from harness.cli import run_session
    answers = iter(['["pwsh"]', "exit"])
    output = []
    run_session(read=lambda prompt: next(answers), write=output.append)
    assert any("policy_denied" in line for line in output)
