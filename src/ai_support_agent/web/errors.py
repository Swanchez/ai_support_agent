"""Safe translation of application failures into public HTTP responses."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ai_support_agent.exceptions import (
    AuthenticationServiceUnavailableError,
    EmbeddingRequestError,
    InvalidCredentialsError,
    InvalidModelResponseError,
    LlmRequestError,
    OrderNotFoundError,
    OrderCancellationConflictError,
    OrderServiceUnavailableError,
)


class ApiErrorResponse(BaseModel):
    """Public error contract that never includes provider or SDK details."""

    detail: str


CHAT_ERROR_RESPONSES = {
    502: {
        "model": ApiErrorResponse,
        "description": "The LLM provider returned a response outside our contract.",
    },
    503: {
        "model": ApiErrorResponse,
        "description": "A required LLM or embedding dependency is temporarily unavailable.",
    },
}

ORDER_ERROR_RESPONSES = {
    404: {
        "model": ApiErrorResponse,
        "description": "The order was not found or is not visible to this user.",
    },
    503: {
        "model": ApiErrorResponse,
        "description": "The order service is temporarily unavailable.",
    },
    409: {
        "model": ApiErrorResponse,
        "description": "The requested order cancellation conflicts with current state.",
    },
}


def register_exception_handlers(app: FastAPI) -> None:
    """Register only runtime failures; configuration must fail before startup."""

    app.add_exception_handler(LlmRequestError, _service_unavailable)
    app.add_exception_handler(EmbeddingRequestError, _service_unavailable)
    app.add_exception_handler(InvalidModelResponseError, _invalid_provider_response)
    app.add_exception_handler(OrderNotFoundError, _order_not_found)
    app.add_exception_handler(OrderServiceUnavailableError, _service_unavailable)
    app.add_exception_handler(OrderCancellationConflictError, _order_cancellation_conflict)
    app.add_exception_handler(AuthenticationServiceUnavailableError, _service_unavailable)
    app.add_exception_handler(InvalidCredentialsError, _invalid_credentials)


def _service_unavailable(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Hide transient provider failures behind one stable public response."""

    _ = request, error
    return _error_response(
        status_code=503,
        detail="Сервис временно недоступен. Попробуйте позже.",
    )


def _invalid_provider_response(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Hide malformed LLM output while preserving the correct failure category."""

    _ = request, error
    return _error_response(
        status_code=502,
        detail="Не удалось обработать ответ сервиса. Попробуйте позже.",
    )


def _order_not_found(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Do not distinguish a missing order from a foreign order."""

    _ = request, error
    return _error_response(status_code=404, detail="Order was not found.")


def _invalid_credentials(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Use one response for unknown logins and incorrect passwords."""

    _ = request, error
    return _error_response(status_code=401, detail="Invalid login or password.")


def _order_cancellation_conflict(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """Do not expose extra internal state or idempotency-record details."""

    _ = request, error
    return _error_response(status_code=409, detail="Order cancellation cannot be completed.")


def _error_response(*, status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ApiErrorResponse(detail=detail).model_dump(),
    )
