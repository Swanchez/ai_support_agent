"""Development-only check of a persistent, idempotent order cancellation."""

from sqlalchemy.exc import SQLAlchemyError

from ai_support_agent.config import load_database_config
from ai_support_agent.persistence.database import (
    create_database_engine,
    create_session_factory,
)
from ai_support_agent.persistence.order_repository import PostgresOrderRepository
from ai_support_agent.tools.order_status import (
    IdempotencyKeyConflict,
    OrderCancellationNotAllowed,
    OrderRepositoryUnavailable,
)


DEMO_USER_ID = "demo-user-1"
DEMO_ORDER_ID = "ORD-1003"
DEMO_IDEMPOTENCY_KEY = "demo-cancel-ord-1003-v1"


def main() -> None:
    """Cancel one seeded order, then make repeat launches safe through one fixed key."""

    engine = create_database_engine(load_database_config())
    session_factory = create_session_factory(engine)
    repository = PostgresOrderRepository(session_factory)
    try:
        result = repository.cancel_visible_to(
            order_id=DEMO_ORDER_ID,
            user_id=DEMO_USER_ID,
            idempotency_key=DEMO_IDEMPOTENCY_KEY,
        )
    except OrderCancellationNotAllowed as error:
        raise SystemExit(f"Demo order cannot be cancelled: {error}") from error
    except IdempotencyKeyConflict as error:
        raise SystemExit(f"Demo idempotency key conflict: {error}") from error
    except (OrderRepositoryUnavailable, SQLAlchemyError) as error:
        raise SystemExit("Could not cancel the demo order in PostgreSQL.") from error
    finally:
        engine.dispose()

    if result is None:
        raise SystemExit("Demo order was not found or does not belong to the demo user.")

    print(
        "Persistent cancellation completed: "
        f"order={result.order_id}; status={result.status.value}; "
        f"idempotency_key={DEMO_IDEMPOTENCY_KEY}"
    )


if __name__ == "__main__":
    main()
