"""Authenticated chat composition for the browser support assistant."""

from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ai_support_agent.agent_assistant_service import AgentAssistantService
from ai_support_agent.agents.core import AgentConversationMessage
from ai_support_agent.agents.runtime import build_gemini_agent_runner
from ai_support_agent.config import GeminiConfig
from ai_support_agent.conversation_service import ConversationService
from ai_support_agent.factory import create_gemini_confirmation_resolver
from ai_support_agent.persistence.audit_sink import PostgresAuditSink
from ai_support_agent.persistence.tool_runtime import create_persistent_tool_executor
from ai_support_agent.rag.retriever import Retriever
from ai_support_agent.service import AnswerResult
from ai_support_agent.tools.audit import BestEffortAuditSink
from ai_support_agent.tools.confirmation import InMemoryPendingActionStore
from ai_support_agent.tools.confirmation_resolver import ConfirmationResolver
from ai_support_agent.tools.context import ToolExecutionContext


@dataclass
class AgentChatService:
    """Create a request-scoped agent around shared, application-owned state.

    The agent runner itself is deliberately short-lived: it contains only one
    bounded reasoning run. Pending write proposals are shared by user ID so the
    next browser message can resolve the right confirmation.
    """

    retriever: Retriever
    config: GeminiConfig
    session_factory: Callable[[], Session]
    confirmation_resolver: ConfirmationResolver
    pending_action_store: InMemoryPendingActionStore = field(
        default_factory=InMemoryPendingActionStore
    )

    def answer(
        self,
        user_question: str,
        context: ToolExecutionContext,
        *,
        history: tuple[AgentConversationMessage, ...] = (),
    ) -> AnswerResult:
        """Run one safe agent turn for the authenticated browser identity."""

        executor = create_persistent_tool_executor(self.session_factory)
        assistant = AgentAssistantService(
            agent_runner=build_gemini_agent_runner(
                retriever=self.retriever,
                config=self.config,
                executor=executor,
                context=context,
            ),
            tool_executor=executor,
            tool_context=context,
            pending_action_store=self.pending_action_store,
        )
        conversation = ConversationService(
            assistant_service=assistant,
            confirmation_resolver=self.confirmation_resolver,
        )
        if conversation.has_pending_action():
            return conversation.answer(user_question)
        return assistant.answer(user_question, history=history)


def create_agent_chat_service(
    *,
    retriever: Retriever,
    config: GeminiConfig,
    session_factory: Callable[[], Session],
) -> AgentChatService:
    """Compose one browser chat service with safe persistent audit recording."""

    pending_action_store = InMemoryPendingActionStore(
        audit_sink=BestEffortAuditSink(PostgresAuditSink(session_factory))
    )
    return AgentChatService(
        retriever=retriever,
        config=config,
        session_factory=session_factory,
        confirmation_resolver=create_gemini_confirmation_resolver(),
        pending_action_store=pending_action_store,
    )
