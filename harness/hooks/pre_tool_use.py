"""Узкий PreToolUse адаптер: только решение, никакого выполнения команды."""

import json
import sys
from pathlib import Path
from typing import TextIO

# Позволяет запускать файл по абсолютному пути независимо от cwd Codex.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.command_policy import Decision, check_command


MAX_INPUT_CHARS = 64_000
# Точные строки, без strip/split и без интерпретации shell-синтаксиса.
SUPPORTED_COMMANDS = {
    "git status --short": ("git", "status", "--short"),
    "docker compose ps": ("docker", "compose", "ps"),
}


def deny(reason: str) -> dict:
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }}


def evaluate_event(event: object) -> dict:
    if not isinstance(event, dict) or event.get("hook_event_name") != "PreToolUse":
        return deny("Unsupported hook event.")
    if event.get("tool_name") != "Bash":
        return deny("Unsupported tool in this demo adapter.")
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict) or not isinstance(tool_input.get("command"), str):
        return deny("Missing string command.")
    command = tool_input["command"]
    if command == "echo HARNESS_BLOCK_DEMO":
        return deny("HARNESS_BLOCK_DEMO: harmless demonstration blocked before execution.")
    argv = SUPPORTED_COMMANDS.get(command)
    if argv is None:
        return deny("Unsupported shell string. This demo does not parse scripts or grant approvals.")
    result = check_command(list(argv))
    if result.decision != Decision.ALLOW:
        return deny("Command not allowed by harness policy.")
    # Успех без разрешения в обход штатных проверок Codex и без updatedInput.
    return {}


def main(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    source = sys.stdin if stdin is None else stdin
    target = sys.stdout if stdout is None else stdout
    try:
        raw = source.read(MAX_INPUT_CHARS + 1)
        if len(raw) > MAX_INPUT_CHARS:
            result = deny("Hook input is too large.")
        else:
            result = evaluate_event(json.loads(raw))
    except Exception:
        # Не выводим input/traceback: они могут содержать секреты. Это явный deny,
        # а не необработанная ошибка hook, которая сама по себе не блокирует tool.
        result = deny("Hook input processing failed.")
    target.write(json.dumps(result, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
