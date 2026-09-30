"""HTTP bearer-token parsing that establishes one trusted tool context."""

from collections.abc import Callable
from hmac import compare_digest
from typing import Annotated, Protocol

from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ai_support_agent.exceptions import InvalidAccessTokenError
from ai_support_agent.tools.context import ToolExecutionContext


class AccessTokenVerifier(Protocol):
    """Application boundary capable of turning a signed token into one user ID."""

    def verify_access_token(self, token: str) -> str:
        """Return a trustworthy identity or raise InvalidAccessTokenError."""


_bearer_scheme = HTTPBearer(auto_error=False)
ACCESS_TOKEN_COOKIE_NAME = "support_access_token"
CSRF_TOKEN_COOKIE_NAME = "support_csrf_token"


def create_current_tool_context_dependency(
    token_verifier: AccessTokenVerifier,
) -> Callable[..., ToolExecutionContext]:
    """Create the FastAPI dependency bound to this application's token service."""

    def get_current_tool_context(
        request: Request,
        credentials: Annotated[
            HTTPAuthorizationCredentials | None,
            Depends(_bearer_scheme),
        ] = None,
        cookie_token: Annotated[
            str | None,
            Cookie(alias=ACCESS_TOKEN_COOKIE_NAME),
        ] = None,
    ) -> ToolExecutionContext:
        bearer_token = (
            credentials.credentials
            if credentials is not None and credentials.scheme.lower() == "bearer"
            else None
        )
        token = bearer_token or cookie_token
        if not token:
            raise _authentication_required()
        try:
            user_id = token_verifier.verify_access_token(token)
        except InvalidAccessTokenError:
            raise _authentication_required() from None
        request.state.authentication_via_cookie = bearer_token is None and cookie_token is not None
        return ToolExecutionContext(current_user_id=user_id)

    return get_current_tool_context


def _authentication_required() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication is required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_csrf_protection_dependency(
    current_tool_context: Callable[..., ToolExecutionContext],
) -> Callable[..., None]:
    """Create a double-submit CSRF check for cookie-authenticated write requests."""

    def require_csrf_token(
        request: Request,
        _: Annotated[ToolExecutionContext, Depends(current_tool_context)],
        cookie_token: Annotated[
            str | None,
            Cookie(alias=CSRF_TOKEN_COOKIE_NAME),
        ] = None,
        header_token: Annotated[
            str | None,
            Header(alias="X-CSRF-Token"),
        ] = None,
    ) -> None:
        if not request.state.authentication_via_cookie:
            return
        if not cookie_token or not header_token or not compare_digest(cookie_token, header_token):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="CSRF validation failed.")

    return require_csrf_token
