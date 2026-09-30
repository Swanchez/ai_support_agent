"""Application service for verifying credentials and issuing access tokens."""

from dataclasses import dataclass
from typing import Protocol

from ai_support_agent.exceptions import (
    AuthenticationServiceUnavailableError,
    InvalidCredentialsError,
)
from ai_support_agent.persistence.user_repository import (
    UserCredentials,
    UserRepositoryUnavailable,
)
from ai_support_agent.security.passwords import verify_password
from ai_support_agent.security.tokens import AccessToken, TokenService


class CredentialsRepository(Protocol):
    """Private storage boundary necessary for one password-based login."""

    def find_credentials(self, login: str) -> UserCredentials | None:
        """Return one credential record, if the login exists."""


@dataclass(frozen=True)
class AuthenticationService:
    """Authenticate one account without exposing login existence to the client."""

    user_repository: CredentialsRepository
    token_service: TokenService

    def login(self, login: str, password: str) -> AccessToken:
        """Issue an access token only after successful password verification."""

        try:
            credentials = self.user_repository.find_credentials(login)
        except UserRepositoryUnavailable as error:
            raise AuthenticationServiceUnavailableError(
                "User repository is temporarily unavailable."
            ) from error

        if credentials is None or not verify_password(password, credentials.password_hash):
            raise InvalidCredentialsError("Invalid login credentials.")
        return self.token_service.issue(credentials.user_id)

    def verify_access_token(self, token: str) -> str:
        """Verify one bearer token and return its authenticated user identity."""

        return self.token_service.verify_subject(token)

    @property
    def cookie_secure(self) -> bool:
        """Return whether browser cookies must be restricted to HTTPS transport."""

        return self.token_service.config.cookie_secure

    @property
    def access_token_ttl_seconds(self) -> int:
        """Return the cookie lifetime matching issued access tokens."""

        return self.token_service.config.access_token_ttl_minutes * 60
