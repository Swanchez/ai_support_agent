import json

import pytest

from harness.cli import run_session


def session(answers):
    iterator = iter(answers)
    output = []

    def read(prompt):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError from None

    executor = run_session(read=read, write=output.append)
    return executor, "\n".join(output)


def test_allow_records_command_without_confirmation():
    executor, output = session(['["docker", "compose", "ps"]', "exit"])
    assert executor.commands == [["docker", "compose", "ps"]]
    assert "allow" in output


def test_confirmation_records_saved_command_once():
    executor, output = session(['["docker", "compose", "down", "-v"]', "да", "exit"])
    assert executor.commands == [["docker", "compose", "down", "-v"]]
    assert "ask" in output
    assert "Сохранённые аргументы" in output
    assert executor.directories[0] in output


def test_rejection_does_not_record_command():
    executor, output = session(['["unknown"]', "нет", "exit"])
    assert executor.commands == []
    assert "Запрос отклонён" in output


def test_deny_does_not_ask_for_confirmation():
    executor, output = session(['["pwsh", "-Command", "test"]', "exit"])
    assert executor.commands == []
    assert "deny" in output
    assert "Запрос:" not in output


@pytest.mark.parametrize("text", ["docker compose ps", json.dumps("docker compose ps"), "null"])
def test_invalid_input_does_not_stop_next_valid_command(text):
    executor, _ = session([text, '["git", "status", "--short"]', "exit"])
    assert executor.commands == [["git", "status", "--short"]]


def test_ambiguous_confirmation_is_not_consent():
    executor, output = session(['["unknown"]', "наверное", "нет", "exit"])
    assert executor.commands == []
    assert "Ответ не распознан" in output


def test_eof_during_confirmation_rejects_request():
    executor, output = session(['["unknown"]'])
    assert executor.commands == []
    assert "Запрос отклонён" in output
