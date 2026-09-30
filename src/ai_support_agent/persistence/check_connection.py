"""Manual diagnostic entry point for the local PostgreSQL connection."""

from sqlalchemy.exc import SQLAlchemyError

from ai_support_agent.config import load_database_config
from ai_support_agent.exceptions import ConfigurationError
from ai_support_agent.persistence.database import (
    check_database_connection,
    create_database_engine,
)


def main() -> None:
    """Connect once without exposing credentials or a full driver traceback."""

    try:
        engine = create_database_engine(load_database_config())
        try:
            check_database_connection(engine)
        finally:
            engine.dispose()
    except ConfigurationError as error:
        raise SystemExit(f"Database configuration error: {error}") from error
    except SQLAlchemyError as error:
        raise SystemExit("Could not connect to PostgreSQL.") from error

    print("PostgreSQL connection is available.")


if __name__ == "__main__":
    main()
