from datetime import UTC, datetime, timedelta

import pytest

from ai_support_agent.config import AuthConfig
from ai_support_agent.exceptions import InvalidAccessTokenError
from ai_support_agent.security.passwords import hash_password, verify_password
from ai_support_agent.security.tokens import TokenService


def test_password_hash_verifies_the_right_password_without_storing_plaintext() -> None:
    password_hash = hash_password("correct horse battery staple")

    assert password_hash != "correct horse battery staple"
    assert verify_password("correct horse battery staple", password_hash) is True
    assert verify_password("wrong password", password_hash) is False


def test_token_service_issues_and_verifies_a_subject() -> None:
    service = TokenService(
        AuthConfig(
            jwt_secret="a" * 32,
            issuer="test-suite",
            access_token_ttl_minutes=30,
            cookie_secure=False,
        )
    )

    issued_at = datetime.now(UTC)
    token = service.issue("demo-user-1", now=issued_at)

    assert service.verify_subject(token.value) == "demo-user-1"
    assert token.expires_at == issued_at + timedelta(minutes=30)


def test_token_service_rejects_a_token_signed_with_another_secret() -> None:
    issuer = "test-suite"
    trusted = TokenService(AuthConfig("a" * 32, issuer, 30, False))
    attacker = TokenService(AuthConfig("b" * 32, issuer, 30, False))
    token = attacker.issue("demo-user-1")

    with pytest.raises(InvalidAccessTokenError):
        trusted.verify_subject(token.value)


def test_token_service_rejects_an_expired_token() -> None:
    service = TokenService(AuthConfig("a" * 32, "test-suite", 30, False))
    token = service.issue("demo-user-1", now=datetime.now(UTC) - timedelta(minutes=31))

    with pytest.raises(InvalidAccessTokenError):
        service.verify_subject(token.value)
