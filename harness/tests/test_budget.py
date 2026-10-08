from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from harness.command_policy import Decision
from harness.events import EventName
from harness.runtime import CommandRuntime, ConfirmationNotFoundError, RecordingExecutor


ALLOWED = ["docker", "compose", "ps"]


def test_default_limit_and_separate_policy_result():
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor)
    for _ in range(3):
        assert not runtime.run(ALLOWED).execution_limit_reached
    result = runtime.run(ALLOWED)
    assert result.policy.decision == Decision.ALLOW
    assert result.execution_limit_reached
    assert result.output is None
    assert runtime.used_executions == len(executor.commands) == 3
    events = runtime.trace.for_operation(result.operation_id)
    assert events[-1].event == EventName.EXECUTION_LIMIT_REACHED
    assert EventName.EXECUTION_STARTED not in [event.event for event in events]


def test_deny_waiting_and_reject_do_not_spend_budget():
    runtime = CommandRuntime(RecordingExecutor(), max_executions=1)
    runtime.run(["pwsh"])
    pending = runtime.run(["unknown"]).pending
    assert runtime.used_executions == 0
    runtime.reject(pending.request_id)
    assert runtime.used_executions == 0
    assert not runtime.run(ALLOWED).execution_limit_reached


def test_confirmation_does_not_bypass_spent_budget():
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor, max_executions=1)
    pending = runtime.run(["unknown"]).pending
    runtime.run(ALLOWED)
    result = runtime.confirm(pending.request_id)
    assert result.execution_limit_reached
    assert executor.commands == [ALLOWED]
    with pytest.raises(ConfirmationNotFoundError):
        runtime.confirm(pending.request_id)


def test_confirmed_call_spends_budget():
    runtime = CommandRuntime(RecordingExecutor(), max_executions=1)
    pending = runtime.run(["unknown"]).pending
    assert not runtime.confirm(pending.request_id).execution_limit_reached
    assert runtime.used_executions == 1
    assert runtime.run(ALLOWED).execution_limit_reached


def test_failed_call_spends_budget():
    class FailingExecutor:
        def execute(self, argv, *, cwd):
            raise RuntimeError("Simulated failure")

    runtime = CommandRuntime(FailingExecutor(), max_executions=1)
    with pytest.raises(RuntimeError):
        runtime.run(ALLOWED)
    assert runtime.used_executions == 1
    assert runtime.run(ALLOWED).execution_limit_reached


def test_zero_budget_and_independent_instances():
    first = CommandRuntime(RecordingExecutor(), max_executions=0)
    second = CommandRuntime(RecordingExecutor(), max_executions=1)
    assert first.run(ALLOWED).execution_limit_reached
    assert first.used_executions == 0
    assert not second.run(ALLOWED).execution_limit_reached


@pytest.mark.parametrize("limit", [-1, True, False, 1.5, "3", None])
def test_invalid_limit(limit):
    with pytest.raises(ValueError):
        CommandRuntime(RecordingExecutor(), max_executions=limit)


def test_parallel_runs_cannot_exceed_budget():
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor, max_executions=3)
    barrier = Barrier(12)

    def run_once(_):
        barrier.wait(timeout=5)
        return runtime.run(ALLOWED)

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run_once, range(12)))
    assert sum(not result.execution_limit_reached for result in results) == 3
    assert runtime.used_executions == len(executor.commands) == 3


def test_parallel_confirmations_share_budget():
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor, max_executions=1)
    requests = [runtime.run(["unknown"]).pending.request_id for _ in range(2)]
    barrier = Barrier(2)

    def confirm(request_id):
        barrier.wait(timeout=5)
        return runtime.confirm(request_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(confirm, requests))
    assert sum(result.execution_limit_reached for result in results) == 1
    assert runtime.used_executions == len(executor.commands) == 1


def test_cli_explains_limit():
    from harness.cli import run_session
    answers = iter(['["docker", "compose", "ps"]'] * 4 + ["exit"])
    output = []
    executor = run_session(read=lambda prompt: next(answers), write=output.append)
    assert len(executor.commands) == 3
    assert any("Лимит вызовов исполнителя исчерпан" in line for line in output)
    assert any("execution_limit_reached" in line for line in output)
