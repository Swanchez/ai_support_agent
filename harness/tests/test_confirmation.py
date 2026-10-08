from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from harness.command_policy import Decision, PolicyResult
from harness.runtime import (
    CommandRuntime, ConfirmationNotFoundError, RecordingExecutor,
)


def test_ask_saves_immutable_command_and_directory():
    directory = Path(__file__).resolve().parent
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor)
    argv = ["docker", "compose", "down", "-v"]
    result = runtime.run(argv, cwd=str(directory))
    assert result.pending is not None
    assert result.pending.reason
    assert executor.commands == []
    argv[:] = ["other-command"]

    confirmed = runtime.confirm(result.pending.request_id)
    assert confirmed.policy.decision == Decision.ALLOW
    assert confirmed.output is not None
    assert executor.commands == [["docker", "compose", "down", "-v"]]
    assert executor.directories == [str(directory)]


def test_confirm_cannot_accept_replacement_arguments():
    runtime = CommandRuntime(RecordingExecutor())
    with pytest.raises(TypeError):
        runtime.confirm("id", argv=["other-command"])


def test_confirmation_is_one_use():
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor)
    pending = runtime.run(["unknown"]).pending
    runtime.confirm(pending.request_id)
    with pytest.raises(ConfirmationNotFoundError):
        runtime.confirm(pending.request_id)
    assert executor.commands == [["unknown"]]


def test_rejection_consumes_request_without_execution():
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor)
    pending = runtime.run(["unknown"]).pending
    assert runtime.reject(pending.request_id) == pending
    with pytest.raises(ConfirmationNotFoundError):
        runtime.confirm(pending.request_id)
    assert executor.commands == []


def test_deny_never_creates_confirmation():
    result = CommandRuntime(RecordingExecutor()).run(["pwsh", "-Command", "test"])
    assert result.policy.decision == Decision.DENY
    assert result.pending is None


def test_unknown_request_is_rejected():
    runtime = CommandRuntime(RecordingExecutor())
    with pytest.raises(ConfirmationNotFoundError):
        runtime.confirm("unknown-id")


def test_new_deny_rule_blocks_previously_pending_command(monkeypatch):
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor)
    pending = runtime.run(["unknown"]).pending
    monkeypatch.setattr(
        "harness.runtime.check_command",
        lambda argv: PolicyResult(Decision.DENY, "New prohibition"),
    )
    result = runtime.confirm(pending.request_id)
    assert result.policy.decision == Decision.DENY
    assert executor.commands == []


def test_execution_failure_does_not_make_confirmation_reusable():
    class FailingExecutor:
        def execute(self, argv, *, cwd):
            raise RuntimeError("Simulated failure")

    runtime = CommandRuntime(FailingExecutor())
    pending = runtime.run(["unknown"]).pending
    with pytest.raises(RuntimeError):
        runtime.confirm(pending.request_id)
    with pytest.raises(ConfirmationNotFoundError):
        runtime.confirm(pending.request_id)


def test_concurrent_confirmations_execute_only_once():
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor)
    pending = runtime.run(["unknown"]).pending

    def confirm_once():
        try:
            runtime.confirm(pending.request_id)
            return True
        except ConfirmationNotFoundError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: confirm_once(), range(2))) == [False, True]
    assert executor.commands == [["unknown"]]
