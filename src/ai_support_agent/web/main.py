"""Production composition root for the authenticated FastAPI application."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Callable

from fastapi import FastAPI
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ai_support_agent.config import load_database_config
from ai_support_agent.config import load_auth_config
from ai_support_agent.config import load_gemini_config
from ai_support_agent.persistence.database import (
    create_database_engine,
    create_session_factory,
)
from ai_support_agent.persistence.tool_runtime import create_persistent_tool_executor
from ai_support_agent.persistence.user_repository import PostgresUserRepository
from ai_support_agent.persistence.conversation_repository import PostgresConversationRepository
from ai_support_agent.rag.fallback_retriever import FallbackRetriever
from ai_support_agent.rag.runtime import (
    create_external_reference_gemini_vector_retriever,
    create_gemini_vector_retriever,
)
from ai_support_agent.service import EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD
from ai_support_agent.web.api import create_app
from ai_support_agent.web.chat_service import AgentChatService, create_agent_chat_service
from ai_support_agent.web.conversation_history import ConversationHistoryService
from ai_support_agent.web.order_service import ToolBackedOrderStatusService
from ai_support_agent.web.ui import install_browser_ui
from ai_support_agent.security.authentication import AuthenticationService
from ai_support_agent.security.tokens import TokenService


def build_chat_service(session_factory: Callable[[], Session]) -> AgentChatService:
    """Create the unified browser assistant: RAG, read tools and confirmations."""

    retriever = FallbackRetriever(
        primary=create_gemini_vector_retriever(),
        fallback_factory=create_external_reference_gemini_vector_retriever,
        fallback_threshold=EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD,
    )
    return create_agent_chat_service(
        config=load_gemini_config(),
        retriever=retriever,
        session_factory=session_factory,
    )


def build_database_dependencies() -> tuple[Engine, sessionmaker[Session]]:
    """Create one long-lived PostgreSQL engine and one shared session factory."""

    engine = create_database_engine(load_database_config())
    return engine, create_session_factory(engine)


def build_order_status_service(
    session_factory: Callable[[], Session],
) -> ToolBackedOrderStatusService:
    """Create order tools around the already-created process-wide database pool."""

    executor = create_persistent_tool_executor(session_factory)
    return ToolBackedOrderStatusService(executor)


def build_authentication_service(
    session_factory: Callable[[], Session],
) -> AuthenticationService:
    """Create password and token authentication around the shared database pool."""

    return AuthenticationService(
        user_repository=PostgresUserRepository(session_factory),
        token_service=TokenService(load_auth_config()),
    )


def create_production_app() -> FastAPI:
    """Uvicorn factory that fails fast when required configuration is invalid."""

    engine, session_factory = build_database_dependencies()
    chat_service = build_chat_service(session_factory)
    conversation_history_service = ConversationHistoryService(
        repository=PostgresConversationRepository(session_factory),
        chat_service=chat_service,
    )
    order_status_service = build_order_status_service(session_factory)
    authentication_service = build_authentication_service(session_factory)

    @asynccontextmanager
    async def database_lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Release the process-wide connection pool during FastAPI shutdown."""

        _ = app
        try:
            yield
        finally:
            engine.dispose()

    app = create_app(
        chat_service,
        order_status_service,
        authentication_service=authentication_service,
        conversation_history_service=conversation_history_service,
        lifespan=database_lifespan,
    )
    install_browser_ui(app)
    return app
