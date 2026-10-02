from dataclasses import dataclass, field

import pytest
from sqlalchemy.dialects import postgresql

from ai_support_agent.persistence.seed_demo_orders import (
    DEMO_ORDER_VALUES,
    seed_demo_orders,
)


@dataclass
class FakeScalarResult:
    values: list[str]

    def all(self) -> list[str]:
        return self.values


@dataclass
class FakeResult:
    inserted_ids: list[str]

    def scalars(self) -> FakeScalarResult:
        return FakeScalarResult(self.inserted_ids)


@dataclass
class FakeSession:
    inserted_ids: list[str] = field(
        default_factory=lambda: ["ORD-1001", "ORD-1002", "ORD-1003"]
    )
    statement: object | None = None
    committed: bool = False
    rolled_back: bool = False

    def execute(self, statement: object) -> FakeResult:
        self.statement = statement
        return FakeResult(self.inserted_ids)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def test_seed_demo_orders_uses_one_idempotent_postgresql_insert() -> None:
    session = FakeSession()

    inserted_count = seed_demo_orders(session)  # type: ignore[arg-type]

    sql = str(
        session.statement.compile(dialect=postgresql.dialect())  # type: ignore[union-attr]
    )
    assert inserted_count == 3
    assert session.committed is True
    assert session.rolled_back is False
    assert "INSERT INTO orders" in sql
    assert "ON CONFLICT (id) DO NOTHING" in sql
    assert "RETURNING orders.id" in sql
    assert len(DEMO_ORDER_VALUES) == 4


def test_seed_demo_orders_rolls_back_when_the_insert_fails() -> None:
    @dataclass
    class FailingSession:
        rolled_back: bool = field(default=False)

        def execute(self, statement: object) -> FakeResult:
            _ = statement
            raise RuntimeError("database write failed")

        def commit(self) -> None:
            raise AssertionError("commit must not be called")

        def rollback(self) -> None:
            self.rolled_back = True

    session = FailingSession()

    with pytest.raises(RuntimeError, match="database write failed"):
        seed_demo_orders(session)  # type: ignore[arg-type]

    assert session.rolled_back is True
