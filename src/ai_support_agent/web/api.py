"""FastAPI HTTP adapter for the support assistant application service."""

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from secrets import token_urlsafe
from typing import Annotated, Literal, Protocol

from fastapi import Depends, FastAPI, Header, Path, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai_support_agent.schemas import SupportResponse
from ai_support_agent.service import AnswerResult
from ai_support_agent.web.errors import CHAT_ERROR_RESPONSES, register_exception_handlers
from ai_support_agent.web.errors import ORDER_ERROR_RESPONSES
from ai_support_agent.web.observability import install_request_observability
from ai_support_agent.web.auth import create_current_tool_context_dependency
from ai_support_agent.web.auth import (
    ACCESS_TOKEN_COOKIE_NAME,
    CSRF_TOKEN_COOKIE_NAME,
    create_csrf_protection_dependency,
)
from ai_support_agent.security.authentication import AuthenticationService
from ai_support_agent.security.tokens import AccessToken
from ai_support_agent.tools.context import ToolExecutionContext
from ai_support_agent.tools.order_status import OrderStatusData
from ai_support_agent.web.order_service import ToolBackedOrderStatusService
from ai_support_agent.web.conversation_history import ConversationHistoryService
from ai_support_agent.persistence.conversation_repository import ConversationSummary, StoredChatMessage


class ChatRequest(BaseModel):
    """Public input contract for one support question."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(max_length=4_000)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message must not be blank.")
        return value


class HealthResponse(BaseModel):
    """Minimal readiness response that does not call external dependencies."""

    status: Literal["ok"] = "ok"


class LoginRequest(BaseModel):
    """Strict public credentials contract for one password-based login request."""

    model_config = ConfigDict(extra="forbid")

    login: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class AccessTokenResponse(BaseModel):
    """Public bearer-token response; never contains user credentials or token claims."""

    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: datetime


class LoginResponse(BaseModel):
    """Minimal browser-login response; the credential itself is an HttpOnly cookie."""

    status: Literal["authenticated"] = "authenticated"


class BrowserSessionResponse(BaseModel):
    """Safe identity data needed to render the signed-in browser interface."""

    user_id: str


class ConversationResponse(BaseModel):
    id: str
    title: str
    updated_at: datetime


class ChatMessageResponse(BaseModel):
    id: int
    role: Literal["user", "assistant"]
    content: str
    created_at: datetime


class ChatService(Protocol):
    """Application boundary required by the HTTP layer."""

    def answer(self, user_question: str, context: ToolExecutionContext) -> AnswerResult:
        """Return one validated support answer."""


class OrderStatusService(Protocol):
    """Application boundary for one authenticated order-status lookup."""

    def get_status(self, order_id: str, context: ToolExecutionContext) -> OrderStatusData:
        """Return data only for an order visible to this identity."""


class OrderCancellationService(Protocol):
    """Application boundary for one explicitly confirmed, idempotent cancellation."""

    def cancel(self, order_id: str, context: ToolExecutionContext) -> OrderStatusData:
        """Cancel one visible eligible order and return its final state."""


def create_app(
    chat_service: ChatService,
    order_status_service: OrderStatusService | None = None,
    *,
    authentication_service: AuthenticationService,
    conversation_history_service: ConversationHistoryService | None = None,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
) -> FastAPI:
    """Create the web application around an already configured service."""

    app = FastAPI(title="AI Support Agent", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)
    install_request_observability(app)

    def get_chat_service() -> ChatService:
        return chat_service

    def get_order_status_service() -> OrderStatusService:
        if order_status_service is None:
            raise RuntimeError("Order status service is not configured.")
        return order_status_service

    def get_order_cancellation_service() -> OrderCancellationService:
        if order_status_service is None:
            raise RuntimeError("Order cancellation service is not configured.")
        return order_status_service  # type: ignore[return-value]

    def get_authentication_service() -> AuthenticationService:
        return authentication_service

    def get_conversation_history_service() -> ConversationHistoryService:
        if conversation_history_service is None:
            raise RuntimeError("Conversation history service is not configured.")
        return conversation_history_service

    current_tool_context = create_current_tool_context_dependency(
        get_authentication_service()
    )
    csrf_protection = create_csrf_protection_dependency(current_tool_context)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/api/v1/auth/token", response_model=AccessTokenResponse)
    def create_access_token(
        request: LoginRequest,
        service: Annotated[AuthenticationService, Depends(get_authentication_service)],
    ) -> AccessTokenResponse:
        token: AccessToken = service.login(request.login, request.password)
        return AccessTokenResponse(access_token=token.value, expires_at=token.expires_at)

    @app.post("/api/v1/auth/login", response_model=LoginResponse)
    def login_for_browser(
        request: LoginRequest,
        response: Response,
        service: Annotated[AuthenticationService, Depends(get_authentication_service)],
    ) -> LoginResponse:
        """Establish a browser session without exposing the JWT to page JavaScript."""

        token = service.login(request.login, request.password)
        response.set_cookie(
            key=ACCESS_TOKEN_COOKIE_NAME,
            value=token.value,
            max_age=service.access_token_ttl_seconds,
            httponly=True,
            secure=service.cookie_secure,
            samesite="lax",
            path="/",
        )
        response.set_cookie(
            key=CSRF_TOKEN_COOKIE_NAME,
            value=token_urlsafe(32),
            max_age=service.access_token_ttl_seconds,
            httponly=False,
            secure=service.cookie_secure,
            samesite="lax",
            path="/",
        )
        return LoginResponse()

    @app.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
    def logout_from_browser(
        response: Response,
        service: Annotated[AuthenticationService, Depends(get_authentication_service)],
        _: Annotated[None, Depends(csrf_protection)],
    ) -> Response:
        """Ask the browser to remove its local authentication cookie."""

        response.delete_cookie(
            key=ACCESS_TOKEN_COOKIE_NAME,
            httponly=True,
            secure=service.cookie_secure,
            samesite="lax",
            path="/",
        )
        response.delete_cookie(
            key=CSRF_TOKEN_COOKIE_NAME,
            httponly=False,
            secure=service.cookie_secure,
            samesite="lax",
            path="/",
        )
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    @app.get("/api/v1/auth/session", response_model=BrowserSessionResponse)
    def browser_session(
        context: Annotated[ToolExecutionContext, Depends(current_tool_context)],
    ) -> BrowserSessionResponse:
        """Return the authenticated subject, never the JWT or its raw claims."""

        return BrowserSessionResponse(user_id=context.current_user_id)

    @app.post(
        "/api/v1/chat",
        response_model=SupportResponse,
        responses=CHAT_ERROR_RESPONSES,
    )
    def answer_chat(
        request: ChatRequest,
        service: Annotated[ChatService, Depends(get_chat_service)],
        context: Annotated[ToolExecutionContext, Depends(current_tool_context)],
    ) -> SupportResponse:
        return service.answer(request.message, context).response

    @app.get("/api/v1/conversations", response_model=list[ConversationResponse])
    def list_conversations(
        service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
        context: Annotated[ToolExecutionContext, Depends(current_tool_context)],
    ) -> list[ConversationResponse]:
        return [_conversation_response(item) for item in service.list_for_user(context.current_user_id)]

    @app.post("/api/v1/conversations", response_model=ConversationResponse)
    def create_conversation(
        service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
        context: Annotated[ToolExecutionContext, Depends(current_tool_context)],
        _: Annotated[None, Depends(csrf_protection)],
    ) -> ConversationResponse:
        return _conversation_response(service.create(context.current_user_id))

    @app.get(
        "/api/v1/conversations/{conversation_id}/messages",
        response_model=list[ChatMessageResponse],
    )
    def list_conversation_messages(
        conversation_id: Annotated[str, Path(min_length=36, max_length=36)],
        service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
        context: Annotated[ToolExecutionContext, Depends(current_tool_context)],
    ) -> list[ChatMessageResponse]:
        return [_message_response(item) for item in service.messages(conversation_id, context.current_user_id)]

    @app.post(
        "/api/v1/conversations/{conversation_id}/messages",
        response_model=SupportResponse,
        responses=CHAT_ERROR_RESPONSES,
    )
    def answer_conversation_message(
        conversation_id: Annotated[str, Path(min_length=36, max_length=36)],
        request: ChatRequest,
        service: Annotated[ConversationHistoryService, Depends(get_conversation_history_service)],
        context: Annotated[ToolExecutionContext, Depends(current_tool_context)],
        _: Annotated[None, Depends(csrf_protection)],
    ) -> SupportResponse:
        return service.answer(conversation_id, request.message, context).response

    @app.get(
        "/api/v1/orders/{order_id}",
        response_model=OrderStatusData,
        responses=ORDER_ERROR_RESPONSES,
    )
    def get_order_status_endpoint(
        order_id: Annotated[
            str,
            Path(min_length=6, max_length=32, pattern=r"^ORD-\d+$"),
        ],
        service: Annotated[OrderStatusService, Depends(get_order_status_service)],
        context: Annotated[ToolExecutionContext, Depends(current_tool_context)],
    ) -> OrderStatusData:
        return service.get_status(order_id, context)

    @app.post(
        "/api/v1/orders/{order_id}/cancellation",
        response_model=OrderStatusData,
        responses=ORDER_ERROR_RESPONSES,
    )
    def cancel_order_endpoint(
        order_id: Annotated[
            str,
            Path(min_length=6, max_length=32, pattern=r"^ORD-\d+$"),
        ],
        idempotency_key: Annotated[
            str,
            Header(
                alias="Idempotency-Key",
                min_length=8,
                max_length=128,
                pattern=r"^[A-Za-z0-9._~-]+$",
            ),
        ],
        service: Annotated[
            OrderCancellationService,
            Depends(get_order_cancellation_service),
        ],
        context: Annotated[ToolExecutionContext, Depends(current_tool_context)],
        _: Annotated[None, Depends(csrf_protection)],
    ) -> OrderStatusData:
        """Run the user's final UI-confirmed cancellation command without an LLM call."""

        cancellation_context = ToolExecutionContext(
            current_user_id=context.current_user_id,
            idempotency_key=idempotency_key,
        )
        return service.cancel(order_id, cancellation_context)

    return app


def _conversation_response(value: ConversationSummary) -> ConversationResponse:
    return ConversationResponse(id=value.id, title=value.title, updated_at=value.updated_at)


def _message_response(value: StoredChatMessage) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=value.id,
        role=value.role.value,
        content=value.content,
        created_at=value.created_at,
    )
