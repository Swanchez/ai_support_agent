"""Интерактивная демонстрация: все команды выполняет только RecordingExecutor."""

import json
from collections.abc import Callable
from pathlib import Path

from harness.runtime import CommandRuntime, RecordingExecutor


def show_trace(runtime: CommandRuntime, operation_id: str, write: Callable[[str], None]) -> None:
    write(f"Трасса операции {operation_id}:")
    for event in runtime.trace.for_operation(operation_id):
        details = f" decision={event.decision.value}" if event.decision is not None else ""
        if event.duration_ms is not None:
            details += f" duration_ms={event.duration_ms:.3f}"
        if event.error_type is not None:
            details += f" error_type={event.error_type}"
        write(f"  {event.event.value}{details}")


def run_session(
    read: Callable[[str], str] = input,
    write: Callable[[str], None] = print,
) -> RecordingExecutor:
    executor = RecordingExecutor()
    runtime = CommandRuntime(executor)
    directory = str(Path.cwd().resolve())
    write("Учебный harness: реальные команды НЕ запускаются.")
    write(f"Лимит вызовов исполнителя за сессию: {runtime.max_executions}")
    write(f"Рабочая директория: {directory}")
    write('Введите JSON-список, например ["docker", "compose", "ps"].')
    write("Для завершения напишите: выход")

    while True:
        try:
            text = read("Команда (JSON): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.casefold() in {"выход", "exit"}:
            break
        try:
            argv = json.loads(text)
        except json.JSONDecodeError:
            write("Некорректный JSON. Используйте список аргументов в двойных кавычках.")
            continue

        result = runtime.run(argv, cwd=directory)
        write(f"Решение: {result.policy.decision.value}. {result.policy.reason}")
        if result.pending is not None:
            pending = result.pending
            write(f"Запрос: {pending.request_id}")
            write(f"Сохранённые аргументы: {json.dumps(pending.argv, ensure_ascii=False)}")
            write(f"Директория: {pending.cwd}")
            while True:
                try:
                    answer = read("Подтвердить? да / нет: ").strip().casefold()
                except (EOFError, KeyboardInterrupt):
                    runtime.reject(pending.request_id)
                    write("Запрос отклонён. Завершение демонстрации.")
                    show_trace(runtime, result.operation_id, write)
                    return executor
                if answer in {"да", "yes"}:
                    result = runtime.confirm(pending.request_id)
                    write(f"После подтверждения: {result.policy.decision.value}.")
                    break
                if answer in {"нет", "no", "выход", "exit"}:
                    runtime.reject(pending.request_id)
                    write("Запрос отклонён. Исполнитель не вызван.")
                    if answer in {"выход", "exit"}:
                        show_trace(runtime, result.operation_id, write)
                        return executor
                    break
                write("Ответ не распознан. Команда не выполнялась; введите да или нет.")

        if result.output is not None:
            write(result.output)
        if result.execution_limit_reached:
            write("Лимит вызовов исполнителя исчерпан. Команда не выполнялась.")
        write(f"Бюджет исполнителя: {runtime.used_executions}/{runtime.max_executions}")
        write(f"Записано учебных выполнений: {len(executor.commands)}")
        show_trace(runtime, result.operation_id, write)

    write("Демонстрация завершена.")
    return executor


def main() -> None:
    run_session()


if __name__ == "__main__":
    main()
