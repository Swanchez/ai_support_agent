import pytest
import harness.command_policy as policy

from harness.command_policy import Decision, check_command


@pytest.mark.parametrize("argv", [
    ["docker", "compose", "ps"],
    ["git", "status", "--short"],
])
def test_allow_exact_read_only_commands(argv):
    assert check_command(argv).decision == Decision.ALLOW


@pytest.mark.parametrize("argv", [
    ["docker", "compose", "down", "-v"],
    ["docker", "compose", "down", "--volumes"],
])
def test_volume_deletion_requires_permission(argv):
    result = check_command(argv)
    assert result.decision == Decision.ASK
    assert "volumes" in result.reason


@pytest.mark.parametrize("argv", [
    ["docker", "compose", "ps", "--all"],
    ["docker", "compose", "down", "-vt"],
    ["python", "script.py"],
    ["git", "status", "--short", ";", "other-command"],
    ["C:/tools/pwsh.exe", "-Command", "anything"],
])
def test_unknown_commands_are_not_automatically_allowed(argv):
    assert check_command(argv).decision == Decision.ASK


@pytest.mark.parametrize("argv", [
    [], None, "docker compose ps", [123], [""], ["git", "bad\x00arg"],
    ["pwsh", "-Command", "anything"],
    ["CMD.EXE", "/c", "anything"],
])
def test_invalid_input_and_shell_are_denied(argv):
    assert check_command(argv).decision == Decision.DENY


def test_input_is_not_modified():
    argv = ["docker", "compose", "ps"]
    original = argv.copy()
    check_command(argv)
    assert argv == original


def test_unmatched_rule_returns_none():
    assert policy.forbid_shell(["docker", "compose", "ps"]) is None


def test_rules_are_ordered_by_policy():
    assert policy.RULES == (
        policy.forbid_shell,
        policy.require_volume_deletion_confirmation,
        policy.allow_read_only_command,
    )


def test_first_matching_rule_stops_evaluation(monkeypatch):
    def deny(argv):
        return policy.PolicyResult(Decision.DENY, "blocked")

    def should_not_run(argv):
        pytest.fail("Rules after a matching rule must not run")

    monkeypatch.setattr(policy, "RULES", (deny, should_not_run))
    assert check_command(["test"]).decision == Decision.DENY


def test_invalid_input_never_reaches_rules(monkeypatch):
    def should_not_run(argv):
        pytest.fail("Invalid input must be rejected before rules")

    monkeypatch.setattr(policy, "RULES", (should_not_run,))
    assert check_command([]).decision == Decision.DENY


def test_all_unmatched_rules_result_in_ask(monkeypatch):
    monkeypatch.setattr(policy, "RULES", (lambda argv: None, lambda argv: None))
    assert check_command(["test"]).decision == Decision.ASK
