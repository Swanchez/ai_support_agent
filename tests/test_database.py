from dataclasses import dataclass

from ai_support_agent.config import DatabaseConfig
from ai_support_agent.persistence.database import (
    check_database_connection,
    database_url,
)


def test_database_url_keeps_special_characters_out_of_manual_string_building() -> None:
    url = database_url(
        DatabaseConfig(
            host="127.0.0.1",
            port=5432,
            database="support",
            user="app_user",
            password="p@ss:word/with spaces",
        )
    )

    assert url.drivername == "postgresql+psycopg"
    assert url.username == "app_user"
    assert url.password == "p@ss:word/with spaces"
    assert url.render_as_string(hide_password=True).endswith("@127.0.0.1:5432/support")
    assert "p@ss" not in url.render_as_string(hide_password=True)


@dataclass
class FakeResult:
    def scalar_one(self) -> int:
        return 1


@dataclass
class FakeConnection:
    statement: object | None = None

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args

    def execute(self, statement: object) -> FakeResult:
        self.statement = statement
        return FakeResult()


@dataclass
class FakeEngine:
    connection: FakeConnection

    def connect(self) -> FakeConnection:
        return self.connection


def test_check_database_connection_executes_only_a_minimal_query() -> None:
    connection = FakeConnection()

    check_database_connection(FakeEngine(connection))  # type: ignore[arg-type]

    assert str(connection.statement) == "SELECT 1"
