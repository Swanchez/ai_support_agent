import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi.testclient import TestClient
import pytest

from ai_support_agent.exceptions import (
    InvalidAccessTokenError,
    InvalidCredentialsError,
    InvalidModelResponseError,
    LlmRequestError,
)
from ai_support_agent.exceptions import OrderNotFoundError
from ai_support_agent.web.api import create_app
from ai_support_agent.schemas import AnswerStatus, SupportResponse
from ai_support_agent.service import AnswerResult
from ai_support_agent.security.tokens import AccessToken
from ai_support_agent.tools.order_status import OrderStatus, OrderStatusData


AUTH_HEADERS = {"Authorization": "Bearer valid-test-token"}


@dataclass
class StubChatService:
    received_questions: list[str] = field(default_factory=list)
    received_user_ids: list[str] = field(default_factory=list)

    def answer(self, user_question: str, context: object) -> AnswerResult:
        self.received_questions.append(user_question)
        self.received_user_ids.append(context.current_user_id)  # type: ignore[union-attr]
        return AnswerResult(
            response=SupportResponse(
                status=AnswerStatus.ANSWERED,
                answer="Тестовый ответ.",
                alternative=None,
                recommendations=[],
                sources=["test-source"],
            ),
            model="test-model",
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        )


@dataclass
class FailingChatService:
    error: Exception

    def answer(self, user_question: str, context: object) -> AnswerResult:
        _ = user_question, context
        raise self.error


@dataclass
class StubOrderStatusService:
    received_requests: list[tuple[str, str]] = field(default_factory=list)
    received_cancellations: list[tuple[str, str, str | None]] = field(default_factory=list)

    def get_status(self, order_id: str, context: object) -> OrderStatusData:
        self.received_requests.append((order_id, context.current_user_id))  # type: ignore[union-attr]
        if order_id != "ORD-1001" or context.current_user_id != "demo-user-1":  # type: ignore[union-attr]
            raise OrderNotFoundError("not visible")
        return OrderStatusData(
            order_id="ORD-1001",
            status=OrderStatus.SHIPPED,
            updated_at="2026-09-20",
            estimated_delivery="2026-09-24",
        )

    def cancel(self, order_id: str, context: object) -> OrderStatusData:
        self.received_cancellations.append(
            (order_id, context.current_user_id, context.idempotency_key)  # type: ignore[union-attr]
        )
        if order_id != "ORD-1003" or context.current_user_id != "demo-user-1":  # type: ignore[union-attr]
            raise OrderNotFoundError("not visible")
        return OrderStatusData(
            order_id="ORD-1003",
            status=OrderStatus.CANCELLED,
            updated_at="2026-09-30",
        )


@dataclass(frozen=True)
class StubAuthenticationService:
    cookie_secure: bool = False
    access_token_ttl_seconds: int = 1_800

    def login(self, login: str, password: str) -> AccessToken:
        if login != "demo-user-1" or password != "demo-password-1":
            raise InvalidCredentialsError("invalid credentials")
        return AccessToken(
            value="valid-test-token",
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
        )

    def verify_access_token(self, token: str) -> str:
        if token == "valid-test-token":
            return "demo-user-1"
        if token == "another-user-token":
            return "demo-user-2"
        raise InvalidAccessTokenError("invalid token")


def create_test_app(
    chat_service: object,
    order_service: object | None = None,
) -> object:
    return create_app(  # type: ignore[arg-type]
        chat_service,
        order_service,
        authentication_service=StubAuthenticationService(),
    )


def test_health_does_not_call_the_chat_service() -> None:
    service = StubChatService()
    client = TestClient(create_test_app(service))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["X-Request-ID"]
    UUID(response.headers["X-Request-ID"])
    assert service.received_questions == []


def test_chat_returns_only_the_public_support_response() -> None:
    service = StubChatService()
    client = TestClient(create_test_app(service))

    response = client.post(
        "/api/v1/chat",
        headers=AUTH_HEADERS,
        json={"message": "  Когда придут деньги?  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "answered",
        "answer": "Тестовый ответ.",
        "alternative": None,
        "recommendations": [],
        "sources": ["test-source"],
    }
    assert service.received_questions == ["Когда придут деньги?"]
    assert service.received_user_ids == ["demo-user-1"]


def test_chat_rejects_a_blank_message_before_calling_the_service() -> None:
    service = StubChatService()
    client = TestClient(create_test_app(service))

    response = client.post("/api/v1/chat", headers=AUTH_HEADERS, json={"message": " \n "})

    assert response.status_code == 422
    assert service.received_questions == []


def test_chat_hides_a_temporary_llm_failure_from_the_client() -> None:
    client = TestClient(
        create_test_app(FailingChatService(LlmRequestError("Gemini failed with HTTP 429.")))
    )

    response = client.post(
        "/api/v1/chat",
        headers=AUTH_HEADERS,
        json={"message": "Когда придут деньги?"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Сервис временно недоступен. Попробуйте позже."}
    assert "Gemini" not in response.text
    assert "429" not in response.text


def test_chat_hides_an_invalid_model_response_from_the_client() -> None:
    client = TestClient(
        create_test_app(FailingChatService(InvalidModelResponseError("Invalid JSON from provider.")))
    )

    response = client.post(
        "/api/v1/chat",
        headers=AUTH_HEADERS,
        json={"message": "Когда придут деньги?"},
    )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "Не удалось обработать ответ сервиса. Попробуйте позже."
    }


def test_openapi_documents_chat_runtime_error_contracts() -> None:
    app = create_test_app(StubChatService())

    responses = app.openapi()["paths"]["/api/v1/chat"]["post"]["responses"]

    assert responses["502"]["description"] == (
        "The LLM provider returned a response outside our contract."
    )
    assert responses["503"]["description"] == (
        "A required LLM or embedding dependency is temporarily unavailable."
    )
    assert responses["502"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/ApiErrorResponse"
    }


def test_openapi_documents_bearer_authentication_for_protected_endpoints() -> None:
    app = create_test_app(StubChatService())

    openapi = app.openapi()

    assert openapi["components"]["securitySchemes"]["HTTPBearer"] == {
        "type": "http",
        "scheme": "bearer",
    }
    assert openapi["paths"]["/api/v1/chat"]["post"]["security"] == [
        {"HTTPBearer": []}
    ]


def test_chat_preserves_a_valid_client_request_id() -> None:
    client = TestClient(create_test_app(StubChatService()))
    request_id = "2f7d3b5a-95ef-4fa6-bfc4-6746882eae45"

    response = client.post(
        "/api/v1/chat",
        headers={"X-Request-ID": request_id, **AUTH_HEADERS},
        json={"message": "Когда придут деньги?"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id


def test_http_log_contains_operational_metadata_but_not_question_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="ai_support_agent.web")
    client = TestClient(create_test_app(StubChatService()))
    private_question = "Секретный текст вопроса пользователя"

    response = client.post(
        "/api/v1/chat", headers=AUTH_HEADERS, json={"message": private_question}
    )

    assert response.status_code == 200
    log_messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "ai_support_agent.web"
    ]
    assert len(log_messages) == 1
    assert "method=POST" in log_messages[0]
    assert "path=/api/v1/chat" in log_messages[0]
    assert "status=200" in log_messages[0]
    assert "request_id=" in log_messages[0]
    assert private_question not in log_messages[0]


def test_chat_rejects_a_missing_bearer_token_before_calling_the_service() -> None:
    service = StubChatService()
    client = TestClient(create_test_app(service))

    response = client.post("/api/v1/chat", json={"message": "Вопрос"})

    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication is required."}
    assert service.received_questions == []


def test_chat_rejects_a_user_id_supplied_in_the_untrusted_json_body() -> None:
    service = StubChatService()
    client = TestClient(create_test_app(service))

    response = client.post(
        "/api/v1/chat",
        headers=AUTH_HEADERS,
        json={"message": "Вопрос", "user_id": "another-user"},
    )

    assert response.status_code == 422
    assert service.received_questions == []


def test_login_returns_a_bearer_token_for_valid_credentials() -> None:
    client = TestClient(create_test_app(StubChatService()))

    response = client.post(
        "/api/v1/auth/token",
        json={"login": "demo-user-1", "password": "demo-password-1"},
    )

    assert response.status_code == 200
    assert response.json()["access_token"] == "valid-test-token"
    assert response.json()["token_type"] == "bearer"
    assert "expires_at" in response.json()


def test_login_does_not_expose_whether_a_login_exists() -> None:
    client = TestClient(create_test_app(StubChatService()))

    response = client.post(
        "/api/v1/auth/token",
        json={"login": "unknown", "password": "anything"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid login or password."}


def test_browser_login_sets_an_httponly_cookie_used_by_a_protected_endpoint() -> None:
    service = StubChatService()
    client = TestClient(create_test_app(service))

    login_response = client.post(
        "/api/v1/auth/login",
        json={"login": "demo-user-1", "password": "demo-password-1"},
    )
    chat_response = client.post("/api/v1/chat", json={"message": "Вопрос"})

    assert login_response.status_code == 200
    assert login_response.json() == {"status": "authenticated"}
    assert "httponly" in login_response.headers["set-cookie"].lower()
    assert "samesite=lax" in login_response.headers["set-cookie"].lower()
    assert "support_csrf_token" in login_response.headers["set-cookie"]
    assert chat_response.status_code == 200
    assert service.received_user_ids == ["demo-user-1"]


def test_browser_logout_removes_the_authentication_cookie() -> None:
    client = TestClient(create_test_app(StubChatService()))
    client.post(
        "/api/v1/auth/login",
        json={"login": "demo-user-1", "password": "demo-password-1"},
    )

    csrf_token = client.cookies["support_csrf_token"]
    logout_response = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": csrf_token})
    chat_response = client.post("/api/v1/chat", json={"message": "Вопрос"})

    assert logout_response.status_code == 204
    assert "max-age=0" in logout_response.headers["set-cookie"].lower()
    assert chat_response.status_code == 401


def test_browser_session_returns_only_the_authenticated_user_id() -> None:
    client = TestClient(create_test_app(StubChatService()))
    client.post(
        "/api/v1/auth/login",
        json={"login": "demo-user-1", "password": "demo-password-1"},
    )

    response = client.get("/api/v1/auth/session")

    assert response.status_code == 200
    assert response.json() == {"user_id": "demo-user-1"}


def test_browser_logout_rejects_a_missing_or_mismatched_csrf_token() -> None:
    client = TestClient(create_test_app(StubChatService()))
    client.post(
        "/api/v1/auth/login",
        json={"login": "demo-user-1", "password": "demo-password-1"},
    )

    missing_response = client.post("/api/v1/auth/logout")
    mismatched_response = client.post(
        "/api/v1/auth/logout",
        headers={"X-CSRF-Token": "attacker-value"},
    )

    assert missing_response.status_code == 403
    assert mismatched_response.status_code == 403
    assert missing_response.json() == {"detail": "CSRF validation failed."}


def test_order_status_uses_the_header_identity_and_returns_only_safe_order_data() -> None:
    order_service = StubOrderStatusService()
    client = TestClient(create_test_app(StubChatService(), order_service))

    response = client.get("/api/v1/orders/ORD-1001", headers=AUTH_HEADERS)

    assert response.status_code == 200
    assert response.json() == {
        "order_id": "ORD-1001",
        "status": "shipped",
        "updated_at": "2026-09-20",
        "estimated_delivery": "2026-09-24",
    }
    assert order_service.received_requests == [("ORD-1001", "demo-user-1")]


def test_order_status_hides_whether_an_order_is_missing_or_belongs_to_someone_else() -> None:
    order_service = StubOrderStatusService()
    client = TestClient(create_test_app(StubChatService(), order_service))

    response = client.get(
        "/api/v1/orders/ORD-1001",
        headers={"Authorization": "Bearer another-user-token"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Order was not found."}
    assert order_service.received_requests == [("ORD-1001", "demo-user-2")]


def test_browser_cancellation_requires_csrf_and_passes_an_idempotency_key() -> None:
    order_service = StubOrderStatusService()
    client = TestClient(create_test_app(StubChatService(), order_service))
    client.post(
        "/api/v1/auth/login",
        json={"login": "demo-user-1", "password": "demo-password-1"},
    )
    csrf_token = client.cookies["support_csrf_token"]

    response = client.post(
        "/api/v1/orders/ORD-1003/cancellation",
        headers={
            "X-CSRF-Token": csrf_token,
            "Idempotency-Key": "cancel-ord-1003-v1",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert order_service.received_cancellations == [
        ("ORD-1003", "demo-user-1", "cancel-ord-1003-v1")
    ]


def test_cookie_cancellation_rejects_a_missing_csrf_token() -> None:
    client = TestClient(create_test_app(StubChatService(), StubOrderStatusService()))
    client.post(
        "/api/v1/auth/login",
        json={"login": "demo-user-1", "password": "demo-password-1"},
    )

    response = client.post(
        "/api/v1/orders/ORD-1003/cancellation",
        headers={"Idempotency-Key": "cancel-ord-1003-v1"},
    )

    assert response.status_code == 403


def test_bearer_cancellation_does_not_require_csrf_protection() -> None:
    order_service = StubOrderStatusService()
    client = TestClient(create_test_app(StubChatService(), order_service))

    response = client.post(
        "/api/v1/orders/ORD-1003/cancellation",
        headers={
            **AUTH_HEADERS,
            "Idempotency-Key": "cancel-ord-1003-v1",
        },
    )

    assert response.status_code == 200
    assert order_service.received_cancellations
