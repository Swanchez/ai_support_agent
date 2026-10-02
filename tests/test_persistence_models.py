from ai_support_agent.persistence.models import (
    Base,
    ChatMessageRecord,
    ConversationRecord,
    IdempotencyRecord,
    OrderRecord,
    ReturnRequestRecord,
)


def test_order_record_maps_the_expected_persistent_columns() -> None:
    orders = Base.metadata.tables["orders"]

    assert OrderRecord.__tablename__ == "orders"
    assert set(orders.columns.keys()) == {
        "id",
        "user_id",
        "status",
        "estimated_delivery_at",
        "updated_at",
    }
    assert orders.primary_key.columns.keys() == ["id"]
    assert orders.c.user_id.index is True
    assert orders.c.estimated_delivery_at.nullable is True
    assert orders.c.updated_at.nullable is False


def test_idempotency_record_has_one_unique_key_per_actor_and_operation() -> None:
    records = Base.metadata.tables["idempotency_records"]

    assert IdempotencyRecord.__tablename__ == "idempotency_records"
    assert set(records.columns.keys()) == {
        "actor_id",
        "operation",
        "idempotency_key",
        "request_fingerprint",
        "result_payload",
        "created_at",
        "expires_at",
    }
    assert set(records.primary_key.columns.keys()) == {
        "actor_id",
        "operation",
        "idempotency_key",
    }
    assert records.c.expires_at.index is True
    assert records.c.result_payload.nullable is False


def test_conversation_records_are_owned_by_one_user() -> None:
    conversations = Base.metadata.tables["conversations"]

    assert ConversationRecord.__tablename__ == "conversations"
    assert set(conversations.columns.keys()) == {
        "id",
        "user_id",
        "title",
        "created_at",
        "updated_at",
    }
    assert conversations.primary_key.columns.keys() == ["id"]
    assert conversations.c.user_id.index is True
    assert conversations.c.updated_at.index is True
    assert list(conversations.c.user_id.foreign_keys)[0].target_fullname == "users.id"


def test_chat_messages_belong_to_one_conversation_and_have_a_safe_role() -> None:
    messages = Base.metadata.tables["chat_messages"]

    assert ChatMessageRecord.__tablename__ == "chat_messages"
    assert set(messages.columns.keys()) == {
        "id",
        "conversation_id",
        "role",
        "content",
        "created_at",
    }
    assert messages.c.conversation_id.index is True
    foreign_key = list(messages.c.conversation_id.foreign_keys)[0]
    assert foreign_key.target_fullname == "conversations.id"
    assert foreign_key.ondelete == "CASCADE"
    assert any(constraint.name == "chat_message_role" for constraint in messages.constraints)


def test_return_request_is_one_active_request_for_one_order() -> None:
    requests = Base.metadata.tables["return_requests"]

    assert ReturnRequestRecord.__tablename__ == "return_requests"
    assert set(requests.columns.keys()) == {
        "id", "order_id", "user_id", "reason", "status", "created_at"
    }
    assert requests.c.order_id.unique is True
    assert requests.c.user_id.index is True
    assert any(constraint.name == "return_request_status" for constraint in requests.constraints)
