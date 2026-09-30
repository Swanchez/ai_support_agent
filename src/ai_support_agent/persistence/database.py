"""SQLAlchemy engine and session infrastructure for PostgreSQL."""

from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from ai_support_agent.config import DatabaseConfig


def database_url(config: DatabaseConfig) -> URL:
    """Build a safely escaped PostgreSQL URL from separate configuration fields."""

    return URL.create(
        drivername="postgresql+psycopg",
        username=config.user,
        password=config.password,
        host=config.host,
        port=config.port,
        database=config.database,
    )


def create_database_engine(config: DatabaseConfig) -> Engine:
    """Create one long-lived engine with pooled PostgreSQL connections."""

    return create_engine(database_url(config), pool_pre_ping=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create short-lived unit-of-work sessions without stale cached objects."""

    return sessionmaker(bind=engine, expire_on_commit=False)


def check_database_connection(engine: Engine) -> None:
    """Make one minimal round trip to prove the database accepts SQL requests."""

    with engine.connect() as connection:
        connection.execute(text("SELECT 1")).scalar_one()
