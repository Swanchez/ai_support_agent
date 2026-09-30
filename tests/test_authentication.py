from dataclasses import dataclass

import pytest

from ai_support_agent.config import AuthConfig
from ai_support_agent.exceptions import InvalidCredentialsError
from ai_support_agent.persistence.user_repository import UserCredentials
from ai_support_agent.security.authentication import AuthenticationService
from ai_support_agent.security.passwords import hash_password
from ai_support_agent.security.tokens import TokenService


@dataclass
class StubCredentialsRepository:
    credentials: UserCredentials | None

    def find_credentials(self, login: str) -> UserCredentials | None:
        _ = login
        return self.credentials


def _token_service() -> TokenService:
    return TokenService(AuthConfig("a" * 32, "test-suite", 30, False))


def test_authentication_service_issues_a_token_for_correct_credentials() -> None:
    service = AuthenticationService(
        StubCredentialsRepository(
            UserCredentials("demo-user-1", hash_password("demo-password-1"))
        ),
        _token_service(),
    )

    token = service.login("demo-user-1", "demo-password-1")

    assert _token_service().verify_subject(token.value) == "demo-user-1"


def test_authentication_service_returns_one_generic_error_for_unknown_or_wrong_credentials() -> None:
    unknown_service = AuthenticationService(StubCredentialsRepository(None), _token_service())
    wrong_password_service = AuthenticationService(
        StubCredentialsRepository(UserCredentials("demo-user-1", hash_password("correct"))),
        _token_service(),
    )

    with pytest.raises(InvalidCredentialsError):
        unknown_service.login("unknown", "anything")
    with pytest.raises(InvalidCredentialsError):
        wrong_password_service.login("demo-user-1", "wrong")
