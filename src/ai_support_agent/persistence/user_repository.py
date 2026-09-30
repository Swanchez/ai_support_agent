"""PostgreSQL adapter for private user credentials used during login only."""

from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ai_support_agent.persistence.models import UserRecord


class UserRepositoryUnavailable(RuntimeError):
    """The user credential store cannot be reached safely."""


@dataclass(frozen=True)
class UserCredentials:
    """Private login record; it must not be sent to HTTP clients or LLMs."""

    user_id: str
    password_hash: str


@dataclass(frozen=True)
class PostgresUserRepository:
    """Look up credentials by unique login without leaking account existence."""

    session_factory: Callable[[], Session]

    def find_credentials(self, login: str) -> UserCredentials | None:
        """Return the private credential record for one normalized login."""

        statement = select(UserRecord).where(UserRecord.login == login)
        try:
            with self.session_factory() as session:
                user = session.scalar(statement)
        except SQLAlchemyError as error:
            raise UserRepositoryUnavailable("PostgreSQL user lookup failed.") from error

        if user is None:
            return None
        return UserCredentials(user_id=user.id, password_hash=user.password_hash)
