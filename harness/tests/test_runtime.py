import pytest

from harness.command_policy import Decision
from harness.runtime import CommandRuntime, RecordingExecutor


def test_allowed_command_reaches_recording_executor_once():
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor)
    argv = ["docker", "compose", "ps"]

    result = runtime.run(argv)

    assert result.policy.decision == Decision.ALLOW
    assert result.output is not None
    assert executor.commands == [argv]
    assert executor.commands[0] is not argv


@pytest.mark.parametrize("argv, expected", [
    (["docker", "compose", "down", "-v"], Decision.ASK),
    (["python", "unknown.py"], Decision.ASK),
    (["pwsh", "-Command", "anything"], Decision.DENY),
    ([], Decision.DENY),
    (None, Decision.DENY),
    ("docker compose ps", Decision.DENY),
])
def test_stopped_command_never_calls_executor(argv, expected):
    class MustNotRunExecutor:
        def execute(self, argv):
            pytest.fail("Stopped commands must never reach the executor")

    result = CommandRuntime(MustNotRunExecutor()).run(argv)

    assert result.policy.decision == expected
    assert result.output is None


def test_recorded_command_does_not_change_with_original_input():
    executor = RecordingExecutor()
    argv = ["docker", "compose", "ps"]
    CommandRuntime(executor).run(argv)
    argv.append("--all")
    assert executor.commands == [["docker", "compose", "ps"]]


def test_recording_executors_do_not_share_history():
    first = RecordingExecutor()
    second = RecordingExecutor()
    first.execute(["test"], cwd=".")
    assert second.commands == []
