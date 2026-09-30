from dataclasses import dataclass

from ai_support_agent.persistence.models import UserRecord
from ai_support_agent.persistence.user_repository import PostgresUserRepository


@dataclass
class FakeSession:
    user: UserRecord | None
    statement: object | None = None

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args

    def scalar(self, statement: object) -> UserRecord | None:
        self.statement = statement
        return self.user


def test_user_repository_returns_only_private_credentials_for_a_login() -> None:
    session = FakeSession(
        UserRecord(
            id="demo-user-1",
            login="demo-user-1",
            password_hash="argon2-hash",
        )
    )
    repository = PostgresUserRepository(lambda: session)  # type: ignore[arg-type]

    credentials = repository.find_credentials("demo-user-1")

    assert credentials is not None
    assert credentials.user_id == "demo-user-1"
    assert credentials.password_hash == "argon2-hash"
    assert "users.login" in str(session.statement)


def test_user_repository_does_not_reveal_a_missing_login() -> None:
    repository = PostgresUserRepository(lambda: FakeSession(None))  # type: ignore[arg-type]

    assert repository.find_credentials("unknown") is None
