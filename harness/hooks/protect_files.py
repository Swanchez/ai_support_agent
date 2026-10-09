"""Учебный PreToolUse для apply_patch. Не читает содержимое файлов и не пишет их."""

import json
from pathlib import Path, PureWindowsPath
import sys
from typing import TextIO

MAX_INPUT_CHARS = 64_000
PROTECTED_NAMES = frozenset({".env", "protected-demo.txt"})
HEADERS = ("*** Add File: ", "*** Update File: ", "*** Delete File: ")


def deny(reason: str) -> dict:
    return {"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }}


def patch_paths(patch: str) -> list[str]:
    """Узкий разбор стандартного patch; неизвестный формат не пропускаем."""
    lines = patch.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError("Invalid patch envelope")
    paths = []
    operation = None
    body_started = False
    moved = False
    for line in lines[1:-1]:
        header = next((prefix for prefix in HEADERS if line.startswith(prefix)), None)
        if header:
            paths.append(line[len(header):])
            operation = header
            body_started = moved = False
        elif line.startswith("*** Move to: "):
            if operation != HEADERS[1] or body_started or moved:
                raise ValueError("Unexpected move")
            paths.append(line[len("*** Move to: "):])
            moved = True
        elif operation == HEADERS[0] and line.startswith("+"):
            body_started = True
        elif operation == HEADERS[1] and (
            line.startswith((" ", "+", "-", "@@ "))
            or line in {"@@", "*** End of File"}
        ):
            body_started = True
        else:
            raise ValueError("Unsupported patch syntax")
    if not paths:
        raise ValueError("No file operations")
    return paths


def check_path(raw: str, cwd: Path) -> None:
    # Консервативно отклоняем неоднозначные пути, ADS и выход из cwd.
    # Проверяем также исходное написание: symlink с именем .env не обходит запрет.
    if not raw or raw != raw.strip() or any(ord(char) < 32 for char in raw):
        raise ValueError("Invalid path")
    portable_parts = raw.replace("\\", "/").split("/")
    if ".." in portable_parts or any(part.endswith((".", " ")) for part in portable_parts if part != "."):
        raise ValueError("Ambiguous path")
    windows_path = PureWindowsPath(raw)
    if windows_path.drive and sys.platform != "win32":
        raise ValueError("Foreign absolute path")
    # Двоеточие допустимо только в обычном Windows drive prefix, не в ADS/URI.
    remainder = raw[len(windows_path.drive):] if sys.platform == "win32" else raw
    if ":" in remainder or windows_path.drive.startswith("\\\\"):
        raise ValueError("Unsupported path")
    path = Path(raw)
    if windows_path.drive and not path.is_absolute():
        raise ValueError("Drive-relative path")
    resolved = (path if path.is_absolute() else cwd / path).resolve()
    if not resolved.is_relative_to(cwd):
        raise ValueError("Path outside cwd")
    if any(part.casefold() in PROTECTED_NAMES for part in portable_parts + list(resolved.parts)):
        raise PermissionError("Protected file")


def evaluate_event(event: object) -> dict:
    try:
        if not isinstance(event, dict) or event.get("hook_event_name") != "PreToolUse" or event.get("tool_name") != "apply_patch":
            return deny("Unsupported event or tool.")
        tool_input = event.get("tool_input")
        patch = tool_input.get("command") if isinstance(tool_input, dict) else None
        raw_cwd = event.get("cwd")
        if not isinstance(patch, str) or len(patch) > MAX_INPUT_CHARS or not isinstance(raw_cwd, str):
            return deny("Missing or oversized patch, or missing cwd.")
        cwd = Path(raw_cwd)
        if not cwd.is_absolute() or not cwd.is_dir():
            return deny("Invalid cwd.")
        cwd = cwd.resolve()
        for path in patch_paths(patch):
            check_path(path, cwd)
    except PermissionError:
        return deny("HARNESS_PROTECTED_FILE: patch targets a protected file.")
    except Exception:
        # Не возвращаем patch, пути или traceback: они могут содержать секреты.
        return deny("Unsupported patch or path; no file changes approved.")
    return {}  # Не выдаём allow в обход штатных разрешений среды.


def main(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    source = sys.stdin if stdin is None else stdin
    target = sys.stdout if stdout is None else stdout
    try:
        raw = source.read(MAX_INPUT_CHARS + 1)
        result = deny("Hook input is too large.") if len(raw) > MAX_INPUT_CHARS else evaluate_event(json.loads(raw))
    except Exception:
        result = deny("Hook input processing failed.")
    target.write(json.dumps(result, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    main()
