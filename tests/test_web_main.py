from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from ai_support_agent.exceptions import ConfigurationError
from ai_support_agent.llm_client import FakeLlmClient
from ai_support_agent.rag.retriever import RetrievedChunk
from ai_support_agent.web import main


@dataclass
class StubRetriever:
    chunks: list[RetrievedChunk]
    received_question: str | None = None

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        _ = top_k, threshold, max_chunks_per_document
        self.received_question = question
        return self.chunks


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


def test_production_app_composes_the_read_only_rag_service(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeLlmClient()
    primary = StubRetriever([])
    fallback = StubRetriever([])
    monkeypatch.setattr(main, "create_llm_client", lambda: client)
    monkeypatch.setattr(main, "create_gemini_vector_retriever", lambda: primary)
    monkeypatch.setattr(
        main,
        "create_external_reference_gemini_vector_retriever",
        lambda: fallback,
    )
    monkeypatch.setattr(main, "build_database_dependencies", lambda: (FakeEngine(), object()))
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
    assert response.json()["status"] == "insufficient_context"
    assert primary.received_question == "Где заказ?"
    assert fallback.received_question == "Где заказ?"


def test_production_app_fails_fast_for_invalid_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_to_create_client() -> FakeLlmClient:
        raise ConfigurationError("GEMINI_API_KEY is not configured.")

    monkeypatch.setattr(main, "create_llm_client", fail_to_create_client)
    monkeypatch.setattr(main, "create_gemini_vector_retriever", lambda: StubRetriever([]))

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        main.create_production_app()
