"""Explicit development-only seed data for the local PostgreSQL database."""

from datetime import UTC, datetime

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ai_support_agent.config import load_database_config
from ai_support_agent.persistence.database import (
    create_database_engine,
    create_session_factory,
)
from ai_support_agent.persistence.models import OrderRecord
from ai_support_agent.tools.order_status import OrderStatus


DEMO_ORDER_VALUES = (
    {
        "id": "ORD-1001",
        "user_id": "demo-user-1",
        "status": OrderStatus.SHIPPED.value,
        "estimated_delivery_at": datetime(2026, 9, 24, tzinfo=UTC).date(),
        "updated_at": datetime(2026, 9, 20, tzinfo=UTC),
    },
    {
        "id": "ORD-1002",
        "user_id": "demo-user-2",
        "status": OrderStatus.PROCESSING.value,
        "estimated_delivery_at": None,
        "updated_at": datetime(2026, 9, 21, tzinfo=UTC),
    },
    {
        "id": "ORD-1003",
        "user_id": "demo-user-1",
        "status": OrderStatus.PACKED.value,
        "estimated_delivery_at": None,
        "updated_at": datetime(2026, 9, 22, tzinfo=UTC),
    },
    {
        "id": "ORD-1004",
        "user_id": "demo-user-1",
        "status": OrderStatus.DELIVERED.value,
        "estimated_delivery_at": datetime(2026, 9, 18, tzinfo=UTC).date(),
        "updated_at": datetime(2026, 9, 18, tzinfo=UTC),
    },
)


def seed_demo_orders(session: Session) -> int:
    """Insert deterministic demo orders once, without overwriting existing rows."""

    statement = (
        insert(OrderRecord)
        .values(DEMO_ORDER_VALUES)
        .on_conflict_do_nothing(index_elements=[OrderRecord.id])
        .returning(OrderRecord.id)
    )
    try:
        inserted_ids = session.execute(statement).scalars().all()
        session.commit()
    except Exception:
        session.rollback()
        raise
    return len(inserted_ids)


def main() -> None:
    """Seed the running local database without showing connection credentials."""

    engine = create_database_engine(load_database_config())
    session_factory = create_session_factory(engine)
    try:
        with session_factory() as session:
            inserted_count = seed_demo_orders(session)
    except SQLAlchemyError as error:
        raise SystemExit("Could not seed demo orders.") from error
    finally:
        engine.dispose()

    print(f"Demo order seed completed; inserted={inserted_count}.")


if __name__ == "__main__":
    main()
