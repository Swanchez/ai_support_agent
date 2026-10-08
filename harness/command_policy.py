"""Проверка структурированной команды. Ничего не запускает."""

from dataclasses import dataclass
from enum import StrEnum
from collections.abc import Callable


class Decision(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyResult:
    decision: Decision
    reason: str


# Разрешены только эти точные команды, без дополнительных аргументов.
READ_ONLY_COMMANDS = frozenset({
    ("docker", "compose", "ps"),
    ("git", "status", "--short"),
})
SHELL_EXECUTABLES = frozenset({
    "powershell", "powershell.exe", "pwsh", "pwsh.exe",
    "cmd", "cmd.exe", "bash", "sh",
})


def validate_arguments(argv: list[str]) -> PolicyResult | None:
    """Отсеять некорректный вход до обращения правил к argv[0]."""
    if not isinstance(argv, list) or not argv:
        return PolicyResult(Decision.DENY, "Нужен непустой список аргументов.")
    if any(not isinstance(arg, str) or not arg or "\x00" in arg for arg in argv):
        return PolicyResult(Decision.DENY, "Некорректные аргументы команды.")
    return None


def forbid_shell(argv: list[str]) -> PolicyResult | None:
    """Правила получают только уже проверенный список аргументов."""
    if argv[0].casefold() in SHELL_EXECUTABLES:
        return PolicyResult(Decision.DENY, "Запуск shell запрещён учебной политикой.")
    return None


def require_volume_deletion_confirmation(argv: list[str]) -> PolicyResult | None:
    if argv[:3] == ["docker", "compose", "down"] and any(
        arg in {"-v", "--volumes"} for arg in argv[3:]
    ):
        return PolicyResult(
            Decision.ASK,
            "Удаление volumes требует отдельного разрешения на эту команду.",
        )
    return None


def allow_read_only_command(argv: list[str]) -> PolicyResult | None:
    if tuple(argv) in READ_ONLY_COMMANDS:
        return PolicyResult(Decision.ALLOW, "Команда входит в точный allowlist.")
    return None


# Первое подходящее правило завершает проверку. Порядок — часть политики:
# запреты -> отдельное согласование -> точные разрешения.
Rule = Callable[[list[str]], PolicyResult | None]
RULES: tuple[Rule, ...] = (
    forbid_shell,
    require_volume_deletion_confirmation,
    allow_read_only_command,
)


def check_command(argv: list[str]) -> PolicyResult:
    """Решение для argv, а не для произвольной строки shell.

    ASK не означает согласие: вызывающая среда должна остановить выполнение.
    Даже ALLOW не заменяет sandbox и проверки рабочей директории.
    """
    invalid = validate_arguments(argv)
    if invalid is not None:
        return invalid

    for rule in RULES:
        result = rule(argv)
        if result is not None:
            return result

    return PolicyResult(Decision.ASK, "Команда не входит в allowlist: нужна проверка.")
