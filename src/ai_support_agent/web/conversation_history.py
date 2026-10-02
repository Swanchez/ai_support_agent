"""Application service for private durable browser conversations."""

from dataclasses import dataclass, replace
from typing import Protocol

from ai_support_agent.agents.core import AgentConversationMessage
from ai_support_agent.exceptions import ConversationNotFoundError
from ai_support_agent.persistence.conversation_repository import (
    DEFAULT_LLM_CONTEXT_MESSAGE_LIMIT,
    DEFAULT_VISIBLE_MESSAGE_LIMIT,
    ChatMessageRole,
    ConversationSummary,
    PostgresConversationRepository,
    StoredChatMessage,
)
from ai_support_agent.service import AnswerResult
from ai_support_agent.tools.context import ToolExecutionContext


class HistoryAwareChatService(Protocol):
    def answer(
        self,
        user_question: str,
        context: ToolExecutionContext,
        *,
        history: tuple[AgentConversationMessage, ...] = (),
    ) -> AnswerResult: ...


@dataclass(frozen=True)
class ConversationHistoryService:
    """Own dialogue lifecycle and retain a small, safe LLM context window."""

    repository: PostgresConversationRepository
    chat_service: HistoryAwareChatService

    def create(self, user_id: str) -> ConversationSummary:
        return self.repository.create_conversation(user_id, "Новый диалог")

    def list_for_user(self, user_id: str) -> list[ConversationSummary]:
        self.repository.delete_expired()
        return self.repository.list_visible_to(user_id)

    def messages(self, conversation_id: str, user_id: str) -> list[StoredChatMessage]:
        messages = self.repository.list_recent_visible_messages(
            conversation_id, user_id, limit=DEFAULT_VISIBLE_MESSAGE_LIMIT
        )
        if messages is None:
            raise ConversationNotFoundError("Conversation is not visible.")
        return messages

    def answer(
        self, conversation_id: str, user_message: str, context: ToolExecutionContext
    ) -> AnswerResult:
        prior = self.repository.list_recent_visible_messages(
            conversation_id,
            context.current_user_id,
            limit=DEFAULT_LLM_CONTEXT_MESSAGE_LIMIT,
        )
        if prior is None:
            raise ConversationNotFoundError("Conversation is not visible.")
        self.repository.append_visible_message(
            conversation_id, context.current_user_id, ChatMessageRole.USER, user_message
        )
        result = self.chat_service.answer(
            user_message,
            replace(context, conversation_id=conversation_id),
            history=tuple(
                AgentConversationMessage(message.role.value, message.content)
                for message in prior
            ),
        )
        self.repository.append_visible_message(
            conversation_id,
            context.current_user_id,
            ChatMessageRole.ASSISTANT,
            _display_text(result),
        )
        return result


def _display_text(result: AnswerResult) -> str:
    response = result.response
    parts = [response.answer]
    if response.alternative:
        parts.append(response.alternative)
    if response.recommendations:
        parts.append("Рекомендации:\n" + "\n".join(f"• {item}" for item in response.recommendations))
    return "\n\n".join(parts)
