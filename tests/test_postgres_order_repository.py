from dataclasses import dataclass
from datetime import UTC, date, datetime

import pytest
from sqlalchemy.exc import OperationalError

from ai_support_agent.persistence.models import IdempotencyRecord, OrderRecord
from ai_support_agent.persistence.order_repository import (
    PostgresOrderRepository,
    cancellation_request_fingerprint,
)
from ai_support_agent.tools.order_status import (
    IdempotencyKeyConflict,
    OrderRepositoryUnavailable,
    OrderStatus,
)


@dataclass
class FakeSession:
    record: OrderRecord | None
    statement: object | None = None

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        _ = args

    def scalar(self, statement: object) -> OrderRecord | None:
        self.statement = statement
        return self.record


def test_postgres_repository_filters_by_order_and_owner_and_maps_safe_data() -> None:
    session = FakeSession(
        OrderRecord(
            id="ORD-1001",
            user_id="demo-user-1",
            status="shipped",
            estimated_delivery_at=date(2026, 9, 24),
            updated_at=datetime(2026, 9, 20, tzinfo=UTC),
        )
    )
    repository = PostgresOrderRepository(lambda: session)  # type: ignore[arg-type]

    order = repository.find_visible_to("ORD-1001", "demo-user-1")

    assert order is not None
    assert order.order_id == "ORD-1001"
    assert order.status is OrderStatus.SHIPPED
    assert order.updated_at == "2026-09-20"
    assert order.estimated_delivery == "2026-09-24"
    assert "orders.id = :id_1" in str(session.statement)
    assert "orders.user_id = :user_id_1" in str(session.statement)


def test_postgres_repository_returns_none_without_revealing_nonvisible_orders() -> None:
    repository = PostgresOrderRepository(lambda: FakeSession(None))  # type: ignore[arg-type]

    assert repository.find_visible_to("ORD-1002", "demo-user-1") is None


def test_postgres_repository_lists_only_owned_orders() -> None:
    first = OrderRecord(
        id="ORD-1003",
        user_id="demo-user-1",
        status="packed",
        estimated_delivery_at=None,
        updated_at=datetime(2026, 9, 22, tzinfo=UTC),
    )
    second = OrderRecord(
        id="ORD-1001",
        user_id="demo-user-1",
        status="shipped",
        estimated_delivery_at=date(2026, 9, 24),
        updated_at=datetime(2026, 9, 20, tzinfo=UTC),
    )

    @dataclass
    class ScalarsResult:
        records: list[OrderRecord]

        def all(self) -> list[OrderRecord]:
            return self.records

    @dataclass
    class ListSession:
        statement: object | None = None

        def __enter__(self) -> "ListSession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def scalars(self, statement: object) -> ScalarsResult:
            self.statement = statement
            return ScalarsResult([first, second])

    session = ListSession()
    repository = PostgresOrderRepository(lambda: session)  # type: ignore[arg-type]

    orders = repository.list_visible_to("demo-user-1")

    assert [order.order_id for order in orders] == ["ORD-1003", "ORD-1001"]
    assert "orders.user_id = :user_id_1" in str(session.statement)


def test_postgres_repository_wraps_database_failures_in_the_tool_boundary() -> None:
    @dataclass
    class UnavailableSession:
        def __enter__(self) -> "UnavailableSession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def scalar(self, statement: object) -> None:
            _ = statement
            raise OperationalError("SELECT", {}, RuntimeError("connection lost"))

    repository = PostgresOrderRepository(lambda: UnavailableSession())  # type: ignore[arg-type]

    with pytest.raises(OrderRepositoryUnavailable, match="PostgreSQL order lookup"):
        repository.find_visible_to("ORD-1001", "demo-user-1")


def test_cancellation_fingerprint_is_stable_per_order_and_differs_for_another_order() -> None:
    assert cancellation_request_fingerprint("ORD-1003") == cancellation_request_fingerprint(
        "ORD-1003"
    )
    assert cancellation_request_fingerprint("ORD-1003") != cancellation_request_fingerprint(
        "ORD-1004"
    )


def test_postgres_repository_returns_a_matching_saved_idempotency_result() -> None:
    order_id = "ORD-1003"
    record = IdempotencyRecord(
        actor_id="demo-user-1",
        operation="cancel_order",
        idempotency_key="confirmation-1",
        request_fingerprint=cancellation_request_fingerprint(order_id),
        result_payload={
            "order_id": order_id,
            "status": "cancelled",
            "updated_at": "2026-09-22",
            "estimated_delivery": None,
        },
        expires_at=datetime(2026, 9, 30, tzinfo=UTC),
    )

    @dataclass
    class IdempotencySession:
        def __enter__(self) -> "IdempotencySession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def get(self, model: object, key: object) -> IdempotencyRecord:
            _ = model, key
            return record

    repository = PostgresOrderRepository(lambda: IdempotencySession())  # type: ignore[arg-type]

    result = repository.find_cancellation_result(
        "demo-user-1", "confirmation-1", order_id
    )

    assert result is not None
    assert result.status is OrderStatus.CANCELLED


def test_postgres_repository_rejects_a_key_reused_for_another_order() -> None:
    record = IdempotencyRecord(
        actor_id="demo-user-1",
        operation="cancel_order",
        idempotency_key="confirmation-1",
        request_fingerprint=cancellation_request_fingerprint("ORD-1003"),
        result_payload={
            "order_id": "ORD-1003",
            "status": "cancelled",
            "updated_at": "2026-09-22",
            "estimated_delivery": None,
        },
        expires_at=datetime(2026, 9, 30, tzinfo=UTC),
    )

    @dataclass
    class IdempotencySession:
        def __enter__(self) -> "IdempotencySession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def get(self, model: object, key: object) -> IdempotencyRecord:
            _ = model, key
            return record

    repository = PostgresOrderRepository(lambda: IdempotencySession())  # type: ignore[arg-type]

    with pytest.raises(IdempotencyKeyConflict):
        repository.find_cancellation_result("demo-user-1", "confirmation-1", "ORD-1004")


def test_postgres_repository_cancels_owned_order_and_saves_retry_result() -> None:
    order = OrderRecord(
        id="ORD-1003",
        user_id="demo-user-1",
        status="packed",
        estimated_delivery_at=None,
        updated_at=datetime(2026, 9, 22, tzinfo=UTC),
    )

    @dataclass
    class FakeTransaction:
        started: bool = False

        def __enter__(self) -> "FakeTransaction":
            self.started = True
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

    @dataclass
    class CancellationSession:
        statement: object | None = None
        saved_record: IdempotencyRecord | None = None
        transaction: FakeTransaction | None = None

        def __enter__(self) -> "CancellationSession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def begin(self) -> FakeTransaction:
            self.transaction = FakeTransaction()
            return self.transaction

        def get(self, model: object, key: object) -> None:
            _ = model, key
            return None

        def scalar(self, statement: object) -> OrderRecord:
            self.statement = statement
            return order

        def add(self, record: IdempotencyRecord) -> None:
            self.saved_record = record

    session = CancellationSession()
    repository = PostgresOrderRepository(lambda: session)  # type: ignore[arg-type]

    result = repository.cancel_visible_to("ORD-1003", "demo-user-1", "confirmation-1")

    assert result is not None
    assert result.status is OrderStatus.CANCELLED
    assert order.status == "cancelled"
    assert session.transaction is not None and session.transaction.started
    assert "FOR UPDATE" in str(session.statement)
    assert session.saved_record is not None
    assert session.saved_record.request_fingerprint == cancellation_request_fingerprint("ORD-1003")
    assert session.saved_record.result_payload["status"] == "cancelled"


def test_postgres_repository_rejects_cancellation_of_a_shipped_order() -> None:
    order = OrderRecord(
        id="ORD-1001",
        user_id="demo-user-1",
        status="shipped",
        estimated_delivery_at=date(2026, 9, 24),
        updated_at=datetime(2026, 9, 20, tzinfo=UTC),
    )

    @dataclass
    class FakeTransaction:
        def __enter__(self) -> "FakeTransaction":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

    @dataclass
    class CancellationSession:
        def __enter__(self) -> "CancellationSession":
            return self

        def __exit__(self, *args: object) -> None:
            _ = args

        def begin(self) -> FakeTransaction:
            return FakeTransaction()

        def get(self, model: object, key: object) -> None:
            _ = model, key
            return None

        def scalar(self, statement: object) -> OrderRecord:
            _ = statement
            return order

    repository = PostgresOrderRepository(lambda: CancellationSession())  # type: ignore[arg-type]

    from ai_support_agent.tools.order_status import OrderCancellationNotAllowed

    with pytest.raises(OrderCancellationNotAllowed):
        repository.cancel_visible_to("ORD-1001", "demo-user-1", "confirmation-1")
