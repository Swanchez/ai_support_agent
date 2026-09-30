"""Creation and validation of short-lived signed access tokens."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from jwt import InvalidTokenError

from ai_support_agent.config import AuthConfig
from ai_support_agent.exceptions import InvalidAccessTokenError


JWT_ALGORITHM = "HS256"


@dataclass(frozen=True)
class AccessToken:
    """Public token response without exposing its internal JWT claims."""

    value: str
    expires_at: datetime


@dataclass(frozen=True)
class TokenService:
    """Issue and validate tokens bound to one configured application issuer."""

    config: AuthConfig

    def issue(self, user_id: str, *, now: datetime | None = None) -> AccessToken:
        """Create a signed access token with an explicit expiry time."""

        issued_at = now or datetime.now(UTC)
        expires_at = issued_at + timedelta(minutes=self.config.access_token_ttl_minutes)
        value = jwt.encode(
            {
                "sub": user_id,
                "iss": self.config.issuer,
                "iat": issued_at,
                "exp": expires_at,
            },
            self.config.jwt_secret,
            algorithm=JWT_ALGORITHM,
        )
        return AccessToken(value=value, expires_at=expires_at)

    def verify_subject(self, token: str) -> str:
        """Return a trustworthy user ID or reject forged, expired, malformed tokens."""

        try:
            claims = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[JWT_ALGORITHM],
                issuer=self.config.issuer,
                options={"require": ["exp", "iat", "iss", "sub"]},
            )
        except InvalidTokenError as error:
            raise InvalidAccessTokenError("Access token is invalid or expired.") from error

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject or len(subject) > 64:
            raise InvalidAccessTokenError("Access token subject is invalid.")
        return subject
