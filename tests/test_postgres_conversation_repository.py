from dataclasses import dataclass, field
from datetime import UTC, datetime

from ai_support_agent.persistence.conversation_repository import (
    ChatMessageRole,
    PostgresConversationRepository,
)
from ai_support_agent.persistence.models import ChatMessageRecord, ConversationRecord


@dataclass
class FakeTransaction:
    def __enter__(self) -> "FakeTransaction":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args


def test_repository_lists_only_the_current_users_conversations() -> None:
    first = ConversationRecord(
        id="dialogue-1",
        user_id="demo-user-1",
        title="Возврат",
        created_at=datetime(2026, 10, 1, tzinfo=UTC),
        updated_at=datetime(2026, 10, 1, tzinfo=UTC),
    )

    @dataclass
    class ScalarsResult:
        records: list[ConversationRecord]

        def all(self) -> list[ConversationRecord]:
            return self.records

    @dataclass
    class FakeSession:
        statement: object | None = None

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def scalars(self, statement: object) -> ScalarsResult:
            self.statement = statement
            return ScalarsResult([first])

    session = FakeSession()
    repository = PostgresConversationRepository(lambda: session)  # type: ignore[arg-type]

    conversations = repository.list_visible_to("demo-user-1")

    assert [conversation.id for conversation in conversations] == ["dialogue-1"]
    assert "conversations.user_id = :user_id_1" in str(session.statement)


def test_repository_returns_a_bounded_chronological_message_window() -> None:
    newer = ChatMessageRecord(
        id=2,
        conversation_id="dialogue-1",
        role="assistant",
        content="Ответ",
        created_at=datetime(2026, 10, 1, 10, 1, tzinfo=UTC),
    )
    older = ChatMessageRecord(
        id=1,
        conversation_id="dialogue-1",
        role="user",
        content="Вопрос",
        created_at=datetime(2026, 10, 1, 10, 0, tzinfo=UTC),
    )

    @dataclass
    class ScalarsResult:
        records: list[ChatMessageRecord]

        def all(self) -> list[ChatMessageRecord]:
            return self.records

    @dataclass
    class FakeSession:
        statement: object | None = None

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def scalars(self, statement: object) -> ScalarsResult:
            self.statement = statement
            return ScalarsResult([newer, older])

    session = FakeSession()
    repository = PostgresConversationRepository(lambda: session)  # type: ignore[arg-type]

    messages = repository.list_recent_visible_messages(
        "dialogue-1", "demo-user-1", limit=12
    )

    assert messages is not None
    assert [message.role for message in messages] == [
        ChatMessageRole.USER,
        ChatMessageRole.ASSISTANT,
    ]
    statement = str(session.statement)
    assert "conversations.user_id = :user_id_1" in statement
    assert "LIMIT :param_1" in statement


def test_repository_creates_a_conversation_with_a_compact_title() -> None:
    @dataclass
    class FakeSession:
        added: list[ConversationRecord] = field(default_factory=list)

        def __enter__(self) -> "FakeSession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def begin(self) -> FakeTransaction:
            return FakeTransaction()

        def add(self, record: ConversationRecord) -> None:
            self.added.append(record)

        def flush(self) -> None:
            self.added[0].created_at = datetime(2026, 10, 1, tzinfo=UTC)
            self.added[0].updated_at = datetime(2026, 10, 1, tzinfo=UTC)

    session = FakeSession()
    repository = PostgresConversationRepository(lambda: session)  # type: ignore[arg-type]

    conversation = repository.create_conversation("demo-user-1", "  Возврат\n  товара ")

    assert conversation.title == "Возврат товара"
    assert session.added[0].user_id == "demo-user-1"
    assert len(conversation.id) == 36
