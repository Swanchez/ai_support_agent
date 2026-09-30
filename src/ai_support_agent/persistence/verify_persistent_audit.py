"""Read one demo order and prove that its safe audit record reaches PostgreSQL."""

from sqlalchemy.exc import SQLAlchemyError

from ai_support_agent.config import ConfigurationError, load_database_config
from ai_support_agent.persistence.database import (
    create_database_engine,
    create_session_factory,
)
from ai_support_agent.persistence.tool_runtime import create_persistent_tool_executor
from ai_support_agent.tools.context import ToolExecutionContext


def main() -> None:
    """Execute a read-only tool with a trusted demo identity."""

    try:
        engine = create_database_engine(load_database_config())
    except ConfigurationError as error:
        raise SystemExit(f"Database configuration error: {error}") from error

    try:
        executor = create_persistent_tool_executor(create_session_factory(engine))
        result = executor.execute(
            "get_order_status",
            {"order_id": "ORD-1001"},
            ToolExecutionContext(current_user_id="demo-user-1"),
        )
    except SQLAlchemyError as error:
        raise SystemExit("Could not execute the persistent demo tool.") from error
    finally:
        engine.dispose()

    print(f"Persistent tool result: {result}")
    print("An audit event should now exist for demo-user-1 and get_order_status.")


if __name__ == "__main__":
    main()
