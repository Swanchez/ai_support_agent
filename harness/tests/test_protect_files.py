import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from harness.hooks.protect_files import MAX_INPUT_CHARS, evaluate_event, main


def event(tmp_path, body):
    return {"hook_event_name": "PreToolUse", "tool_name": "apply_patch",
            "cwd": str(tmp_path), "tool_input": {"command": "*** Begin Patch\n" + body + "\n*** End Patch"}}


def denied(result):
    return result["hookSpecificOutput"]["permissionDecision"] == "deny"


@pytest.mark.parametrize("body", [
    "*** Add File: normal.txt\n+hello",
    "*** Update File: normal.txt\n@@\n-old\n+new",
    "*** Delete File: normal.txt",
    "*** Update File: normal.txt\n*** Move to: renamed.txt\n@@\n-old\n+new",
])
def test_normal_operations_leave_native_permissions(tmp_path, body):
    assert evaluate_event(event(tmp_path, body)) == {}
    assert list(tmp_path.iterdir()) == []  # Hook сам не применяет patch.


@pytest.mark.parametrize("name", [".env", "nested/.env", ".ENV", "protected-demo.txt"])
@pytest.mark.parametrize("operation", ["Add", "Update", "Delete"])
def test_protected_operations_denied(tmp_path, name, operation):
    assert denied(evaluate_event(event(tmp_path, f"*** {operation} File: {name}")))


@pytest.mark.parametrize("source,target", [
    (".env", "normal.txt"), ("normal.txt", ".env"),
    ("protected-demo.txt", "renamed.txt"),
])
def test_move_checks_both_paths(tmp_path, source, target):
    assert denied(evaluate_event(event(tmp_path, f"*** Update File: {source}\n*** Move to: {target}")))


def test_protected_path_blocks_entire_patch(tmp_path):
    body = "*** Add File: normal.txt\n+ok\n*** Delete File: .env"
    assert denied(evaluate_event(event(tmp_path, body)))


@pytest.mark.parametrize("name", ["../outside.txt", ".env.", ".env ", "file:stream", "", "bad\x00path"])
def test_ambiguous_paths_denied(tmp_path, name):
    assert denied(evaluate_event(event(tmp_path, f"*** Delete File: {name}")))


def test_absolute_paths_checked(tmp_path):
    assert evaluate_event(event(tmp_path, f"*** Delete File: {tmp_path / 'normal.txt'}")) == {}
    assert denied(evaluate_event(event(tmp_path, f"*** Delete File: {tmp_path / '.env'}")))
    assert denied(evaluate_event(event(tmp_path, f"*** Delete File: {tmp_path.parent / 'outside.txt'}")))


def test_env_example_is_not_secret_name(tmp_path):
    assert evaluate_event(event(tmp_path, "*** Add File: .env.example\n+PLACEHOLDER=")) == {}


def test_symlink_destination_checked(tmp_path):
    # Только безвредный временный файл. Windows может требовать отдельные права.
    target = tmp_path / "protected-demo.txt"
    target.touch()
    link = tmp_path / "alias.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("Symlinks unavailable")
    assert denied(evaluate_event(event(tmp_path, "*** Delete File: alias.txt")))


@pytest.mark.parametrize("body", ["", "unexpected", "*** Move to: a", "*** Add File: a\n*** Move to: b"])
def test_unknown_patch_format_denied(tmp_path, body):
    assert denied(evaluate_event(event(tmp_path, body)))


@pytest.mark.parametrize("raw", ["not json", "[]", "x" * (MAX_INPUT_CHARS + 1)],
                         ids=["invalid-json", "wrong-shape", "oversized"])
def test_bad_stdin_returns_explicit_deny(raw):
    output = io.StringIO()
    main(io.StringIO(raw), output)
    assert denied(json.loads(output.getvalue()))


def test_unknown_tool_and_missing_cwd_denied(tmp_path):
    value = event(tmp_path, "*** Delete File: normal.txt")
    value["tool_name"] = "Bash"
    assert denied(evaluate_event(value))
    value["tool_name"] = "apply_patch"
    del value["cwd"]
    assert denied(evaluate_event(value))


def test_patch_content_not_echoed(tmp_path):
    value = event(tmp_path, "*** Add File: .env\n+SECRET=demo-only")
    assert "demo-only" not in json.dumps(evaluate_event(value))


def test_standalone_protocol(tmp_path):
    script = Path(__file__).resolve().parents[1] / "hooks" / "protect_files.py"
    result = subprocess.run([sys.executable, "-B", str(script)],
                            input=json.dumps(event(tmp_path, "*** Delete File: protected-demo.txt")),
                            text=True, capture_output=True, cwd=tmp_path, timeout=10, check=True)
    assert result.stderr == ""
    assert denied(json.loads(result.stdout))
