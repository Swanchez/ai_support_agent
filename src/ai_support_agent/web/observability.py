"""Request-scoped identifiers and privacy-conscious HTTP logging."""

import logging
from time import perf_counter
from uuid import UUID, uuid4

from fastapi import FastAPI, Request, Response


LOGGER = logging.getLogger("ai_support_agent.web")
REQUEST_ID_HEADER = "X-Request-ID"


def install_request_observability(app: FastAPI) -> None:
    """Attach one middleware that correlates every HTTP response with its request."""

    @app.middleware("http")
    async def observe_request(request: Request, call_next) -> Response:
        request_id = _request_id_from_header(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        started_at = perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            _log_request(request, request_id, status_code=500, started_at=started_at)
            raise

        response.headers[REQUEST_ID_HEADER] = request_id
        _log_request(
            request,
            request_id,
            status_code=response.status_code,
            started_at=started_at,
        )
        return response


def _request_id_from_header(value: str | None) -> str:
    """Accept only a well-formed UUID to prevent unsafe log values from clients."""

    if value:
        try:
            return str(UUID(value))
        except ValueError:
            pass
    return str(uuid4())


def _log_request(
    request: Request,
    request_id: str,
    *,
    status_code: int,
    started_at: float,
) -> None:
    """Log operational metadata only; never record request or response contents."""

    route = request.scope.get("route")
    path = getattr(route, "path", request.url.path)
    duration_ms = (perf_counter() - started_at) * 1_000
    LOGGER.info(
        "http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        path,
        status_code,
        duration_ms,
    )
