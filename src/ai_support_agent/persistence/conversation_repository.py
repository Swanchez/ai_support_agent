"""PostgreSQL adapter for private browser conversation history."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ai_support_agent.persistence.models import ChatMessageRecord, ConversationRecord


DEFAULT_VISIBLE_MESSAGE_LIMIT = 100
DEFAULT_LLM_CONTEXT_MESSAGE_LIMIT = 8
CONVERSATION_TTL = timedelta(days=1)


class ChatMessageRole(StrEnum):
    """The two actor roles deliberately supported by the browser chat."""

    USER = "user"
    ASSISTANT = "assistant"


class ConversationRepositoryUnavailable(RuntimeError):
    """Conversation data could not be read or written reliably."""


@dataclass(frozen=True)
class ConversationSummary:
    """Safe metadata rendered in the authenticated conversation sidebar."""

    id: str
    title: str
    updated_at: datetime


@dataclass(frozen=True)
class StoredChatMessage:
    """One safe, displayable message owned by a visible conversation."""

    id: int
    role: ChatMessageRole
    content: str
    created_at: datetime


@dataclass(frozen=True)
class PostgresConversationRepository:
    """Store and retrieve only the authenticated user's own conversations."""

    session_factory: Callable[[], Session]

    def create_conversation(self, user_id: str, title: str) -> ConversationSummary:
        """Create a user-owned dialogue with an application-generated opaque ID."""

        record = ConversationRecord(
            id=str(uuid4()),
            user_id=user_id,
            title=_normalize_title(title),
        )
        try:
            with self.session_factory() as session, session.begin():
                session.add(record)
                session.flush()
                return _to_summary(record)
        except SQLAlchemyError as error:
            raise ConversationRepositoryUnavailable(
                "PostgreSQL conversation creation failed."
            ) from error

    def list_visible_to(self, user_id: str) -> list[ConversationSummary]:
        """List a user's conversations from most recently active to oldest."""

        statement = (
            select(ConversationRecord)
            .where(ConversationRecord.user_id == user_id)
            .order_by(ConversationRecord.updated_at.desc(), ConversationRecord.id)
        )
        try:
            with self.session_factory() as session:
                records = session.scalars(statement).all()
        except SQLAlchemyError as error:
            raise ConversationRepositoryUnavailable(
                "PostgreSQL conversation listing failed."
            ) from error
        return [_to_summary(record) for record in records]

    def delete_expired(self, *, now: datetime | None = None) -> int:
        """Remove inactive dialogues; PostgreSQL cascades deletion to messages."""

        cutoff = (now or datetime.now(UTC)) - CONVERSATION_TTL
        try:
            with self.session_factory() as session, session.begin():
                records = session.scalars(
                    select(ConversationRecord).where(ConversationRecord.updated_at < cutoff)
                ).all()
                for record in records:
                    session.delete(record)
                return len(records)
        except SQLAlchemyError as error:
            raise ConversationRepositoryUnavailable(
                "PostgreSQL conversation cleanup failed."
            ) from error

    def list_recent_visible_messages(
        self,
        conversation_id: str,
        user_id: str,
        *,
        limit: int,
    ) -> list[StoredChatMessage] | None:
        """Return the newest bounded message window, chronologically ordered.

        ``None`` means the dialogue is absent or belongs to someone else.  Both
        cases intentionally have the same observable result.
        """

        if limit < 1:
            raise ValueError("Message limit must be positive.")
        statement = (
            select(ChatMessageRecord)
            .join(
                ConversationRecord,
                ChatMessageRecord.conversation_id == ConversationRecord.id,
            )
            .where(
                ConversationRecord.id == conversation_id,
                ConversationRecord.user_id == user_id,
            )
            .order_by(ChatMessageRecord.created_at.desc(), ChatMessageRecord.id.desc())
            .limit(limit)
        )
        try:
            with self.session_factory() as session:
                records = session.scalars(statement).all()
        except SQLAlchemyError as error:
            raise ConversationRepositoryUnavailable(
                "PostgreSQL message lookup failed."
            ) from error

        if not records:
            return [] if self._is_visible(conversation_id, user_id) else None
        return [_to_message(record) for record in reversed(records)]

    def append_visible_message(
        self,
        conversation_id: str,
        user_id: str,
        role: ChatMessageRole,
        content: str,
    ) -> StoredChatMessage | None:
        """Append one message only after atomically checking conversation ownership."""

        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("Chat message content must not be blank.")

        try:
            with self.session_factory() as session, session.begin():
                conversation = session.scalar(
                    select(ConversationRecord)
                    .where(
                        ConversationRecord.id == conversation_id,
                        ConversationRecord.user_id == user_id,
                    )
                    .with_for_update()
                )
                if conversation is None:
                    return None
                if role is ChatMessageRole.USER and conversation.title == "Новый диалог":
                    conversation.title = _normalize_title(normalized_content)
                message = ChatMessageRecord(
                    conversation_id=conversation_id,
                    role=role.value,
                    content=normalized_content,
                )
                conversation.updated_at = datetime.now(UTC)
                session.add(message)
                session.flush()
                return _to_message(message)
        except SQLAlchemyError as error:
            raise ConversationRepositoryUnavailable(
                "PostgreSQL message creation failed."
            ) from error

    def _is_visible(self, conversation_id: str, user_id: str) -> bool:
        statement = select(ConversationRecord.id).where(
            ConversationRecord.id == conversation_id,
            ConversationRecord.user_id == user_id,
        )
        try:
            with self.session_factory() as session:
                return session.scalar(statement) is not None
        except SQLAlchemyError as error:
            raise ConversationRepositoryUnavailable(
                "PostgreSQL conversation visibility lookup failed."
            ) from error


def _normalize_title(value: str) -> str:
    """Derive a compact sidebar title without asking the model to summarize it."""

    normalized = " ".join(value.split())
    if not normalized:
        return "Новый диалог"
    return normalized[:160]


def _to_summary(record: ConversationRecord) -> ConversationSummary:
    return ConversationSummary(
        id=record.id,
        title=record.title,
        updated_at=record.updated_at,
    )


def _to_message(record: ChatMessageRecord) -> StoredChatMessage:
    return StoredChatMessage(
        id=record.id,
        role=ChatMessageRole(record.role),
        content=record.content,
        created_at=record.created_at,
    )
