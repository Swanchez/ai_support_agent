import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from harness.hooks.pre_tool_use import MAX_INPUT_CHARS, evaluate_event, main


def event(command):
    return {"hook_event_name": "PreToolUse", "tool_name": "Bash", "tool_input": {"command": command}}


@pytest.mark.parametrize("command", ["git status --short", "docker compose ps"])
def test_exact_allowlist_passes_without_rewriting(command):
    assert evaluate_event(event(command)) == {}


@pytest.mark.parametrize("command", [
    "echo HARNESS_BLOCK_DEMO", "docker compose down -v",
    "git status --short; echo other", "git status --short\nother",
    "git status --short ", "pwsh -Command anything", "", None,
])
def test_unknown_or_unsafe_format_gets_explicit_deny(command):
    result = evaluate_event(event(command))["hookSpecificOutput"]
    assert result["hookEventName"] == "PreToolUse"
    assert result["permissionDecision"] == "deny"
    assert "updatedInput" not in result


@pytest.mark.parametrize("value", [None, [], {}, {"hook_event_name": "PostToolUse"}])
def test_bad_event_denied(value):
    assert evaluate_event(value)["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize(
    "raw", ["not json", "[]", "x" * (MAX_INPUT_CHARS + 1)],
    ids=["invalid-json", "wrong-shape", "oversized-input"],
)
def test_invalid_stdin_returns_valid_deny_json(raw):
    output = io.StringIO()
    main(io.StringIO(raw), output)
    assert json.loads(output.getvalue())["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_no_secret_input_in_response():
    result = evaluate_event(event("echo secret-token-demo"))
    assert "secret-token-demo" not in json.dumps(result)


def test_standalone_script_from_another_directory():
    script = Path(__file__).resolve().parents[1] / "hooks" / "pre_tool_use.py"
    result = subprocess.run(
        [sys.executable, "-B", str(script)],
        input=json.dumps(event("echo HARNESS_BLOCK_DEMO")),
        text=True, capture_output=True, cwd=script.parent, timeout=10, check=True,
    )
    assert result.stderr == ""
    assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_example_config_is_inactive_template():
    path = Path(__file__).resolve().parents[1] / "hooks" / "hooks.example.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    handler = config["hooks"]["PreToolUse"][0]["hooks"][0]
    assert "__PYTHON_ABSOLUTE_PATH__" in handler["command"]
    assert "__HOOK_SCRIPT_ABSOLUTE_PATH__" in handler["commandWindows"]
