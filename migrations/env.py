"""Alembic runtime configuration for the AI Support Agent PostgreSQL schema."""

from alembic import context
from sqlalchemy import create_engine, pool

from ai_support_agent.config import load_database_config
from ai_support_agent.persistence.database import database_url
from ai_support_agent.persistence.models import Base


target_metadata = Base.metadata


def _database_url() -> str:
    """Build the URL at runtime so credentials never live in alembic.ini."""

    return database_url(load_database_config()).render_as_string(hide_password=False)


def run_migrations_offline() -> None:
    """Generate SQL without opening a PostgreSQL connection."""

    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations inside one database transaction."""

    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )

            with context.begin_transaction():
                context.run_migrations()
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
