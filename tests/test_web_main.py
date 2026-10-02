from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from ai_support_agent.exceptions import ConfigurationError
from ai_support_agent.schemas import AnswerStatus, SupportResponse
from ai_support_agent.service import AnswerResult
from ai_support_agent.web import main


@dataclass
class StubChatService:
    def answer(self, question: str, context: object) -> AnswerResult:
        _ = question, context
        return AnswerResult(
            SupportResponse(
                status=AnswerStatus.ANSWERED,
                answer="Тестовый ответ агента.",
                alternative=None,
                recommendations=[],
                sources=[],
            ),
            "test-agent",
            0,
            0,
            0,
        )


@dataclass
class StubOrderStatusService:
    def get_status(self, order_id: str, context: object) -> None:
        _ = order_id, context
        return None


@dataclass
class FakeEngine:
    disposed: bool = False

    def dispose(self) -> None:
        self.disposed = True


@dataclass(frozen=True)
class StubAuthenticationService:
    def verify_access_token(self, token: str) -> str:
        _ = token
        return "demo-user-1"


def test_production_app_composes_the_unified_agent_chat_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "build_database_dependencies", lambda: (FakeEngine(), object()))
    monkeypatch.setattr(main, "build_chat_service", lambda session_factory: StubChatService())
    monkeypatch.setattr(
        main,
        "build_order_status_service",
        lambda session_factory: StubOrderStatusService(),
    )
    monkeypatch.setattr(
        main,
        "build_authentication_service",
        lambda session_factory: StubAuthenticationService(),
    )

    app = main.create_production_app()
    response = TestClient(app).post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer test-token"},
        json={"message": "Где заказ?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Тестовый ответ агента."


def test_production_app_fails_fast_for_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_create_chat_service(session_factory: object) -> StubChatService:
        _ = session_factory
        raise ConfigurationError("GEMINI_API_KEY is not configured.")

    monkeypatch.setattr(main, "build_database_dependencies", lambda: (FakeEngine(), object()))
    monkeypatch.setattr(main, "build_chat_service", fail_to_create_chat_service)

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        main.create_production_app()
