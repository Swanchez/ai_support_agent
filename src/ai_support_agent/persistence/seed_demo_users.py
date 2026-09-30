"""Explicit development-only users matching the existing seeded demo orders."""

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ai_support_agent.config import load_database_config
from ai_support_agent.persistence.database import (
    create_database_engine,
    create_session_factory,
)
from ai_support_agent.persistence.models import UserRecord
from ai_support_agent.security.passwords import hash_password


DEMO_USERS = (
    {"id": "demo-user-1", "login": "demo-user-1", "password": "demo-password-1"},
    {"id": "demo-user-2", "login": "demo-user-2", "password": "demo-password-2"},
)


def seed_demo_users(session: Session) -> int:
    """Create deterministic local accounts once, storing only salted password hashes."""

    values = [
        {
            "id": user["id"],
            "login": user["login"],
            "password_hash": hash_password(user["password"]),
        }
        for user in DEMO_USERS
    ]
    statement = (
        insert(UserRecord)
        .values(values)
        .on_conflict_do_nothing(index_elements=[UserRecord.id])
        .returning(UserRecord.id)
    )
    try:
        inserted_ids = session.execute(statement).scalars().all()
        session.commit()
    except Exception:
        session.rollback()
        raise
    return len(inserted_ids)


def main() -> None:
    """Seed local-only accounts without printing their password hashes."""

    engine = create_database_engine(load_database_config())
    session_factory = create_session_factory(engine)
    try:
        with session_factory() as session:
            inserted_count = seed_demo_users(session)
    except SQLAlchemyError as error:
        raise SystemExit("Could not seed demo users.") from error
    finally:
        engine.dispose()

    print(f"Demo user seed completed; inserted={inserted_count}.")


if __name__ == "__main__":
    main()
