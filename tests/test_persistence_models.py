from ai_support_agent.persistence.models import Base, IdempotencyRecord, OrderRecord


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
