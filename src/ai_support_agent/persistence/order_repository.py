"""PostgreSQL read adapter for customer-visible order status data."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ai_support_agent.persistence.models import IdempotencyRecord, OrderRecord
from ai_support_agent.tools.order_status import (
    CANCELLABLE_ORDER_STATUSES,
    IdempotencyKeyConflict,
    OrderCancellationNotAllowed,
    OrderRepositoryUnavailable,
    OrderStatus,
    OrderStatusData,
)


CANCEL_ORDER_OPERATION = "cancel_order"
IDEMPOTENCY_RESULT_TTL = timedelta(days=1)


@dataclass(frozen=True)
class PostgresOrderRepository:
    """PostgreSQL implementation of safe reads and cancellation with idempotency."""

    session_factory: Callable[[], Session]

    def find_visible_to(self, order_id: str, user_id: str) -> OrderStatusData | None:
        """Return an order only when both its public ID and owner match the query."""

        statement = select(OrderRecord).where(
            OrderRecord.id == order_id,
            OrderRecord.user_id == user_id,
        )
        try:
            with self.session_factory() as session:
                record = session.scalar(statement)
        except SQLAlchemyError as error:
            raise OrderRepositoryUnavailable("PostgreSQL order lookup failed.") from error

        if record is None:
            return None
        try:
            return _to_order_status_data(record)
        except ValueError as error:
            raise OrderRepositoryUnavailable("Stored order data is invalid.") from error

    def find_cancellation_result(
        self,
        user_id: str,
        idempotency_key: str,
        order_id: str,
    ) -> OrderStatusData | None:
        """Return an earlier matching cancellation result without exposing other data."""

        fingerprint = cancellation_request_fingerprint(order_id)
        try:
            with self.session_factory() as session:
                record = _find_idempotency_record(session, user_id, idempotency_key)
        except SQLAlchemyError as error:
            raise OrderRepositoryUnavailable("PostgreSQL idempotency lookup failed.") from error

        if record is None:
            return None
        return _idempotency_result(record, fingerprint)

    def cancel_visible_to(
        self,
        order_id: str,
        user_id: str,
        idempotency_key: str,
    ) -> OrderStatusData | None:
        """Cancel one owned eligible order and persist its retry result atomically."""

        fingerprint = cancellation_request_fingerprint(order_id)
        try:
            with self.session_factory() as session:
                with session.begin():
                    existing = _find_idempotency_record(session, user_id, idempotency_key)
                    if existing is not None:
                        return _idempotency_result(existing, fingerprint)

                    order = session.scalar(
                        select(OrderRecord)
                        .where(
                            OrderRecord.id == order_id,
                            OrderRecord.user_id == user_id,
                        )
                        .with_for_update()
                    )
                    if order is None:
                        return None

                    # A concurrent request may have committed while this transaction waited.
                    existing = _find_idempotency_record(session, user_id, idempotency_key)
                    if existing is not None:
                        return _idempotency_result(existing, fingerprint)

                    if OrderStatus(order.status) not in CANCELLABLE_ORDER_STATUSES:
                        raise OrderCancellationNotAllowed(
                            "Order cannot be cancelled in its current status."
                        )

                    order.status = OrderStatus.CANCELLED.value
                    order.updated_at = datetime.now(UTC)
                    result = _to_order_status_data(order)
                    session.add(
                        IdempotencyRecord(
                            actor_id=user_id,
                            operation=CANCEL_ORDER_OPERATION,
                            idempotency_key=idempotency_key,
                            request_fingerprint=fingerprint,
                            result_payload=result.model_dump(mode="json"),
                            expires_at=datetime.now(UTC) + IDEMPOTENCY_RESULT_TTL,
                        )
                    )
                    return result
        except SQLAlchemyError as error:
            raise OrderRepositoryUnavailable("PostgreSQL order cancellation failed.") from error
        except ValueError as error:
            raise OrderRepositoryUnavailable("Stored order data is invalid.") from error


def _to_order_status_data(record: OrderRecord) -> OrderStatusData:
    """Map a persistence record to the existing safe tool-data contract."""

    return OrderStatusData(
        order_id=record.id,
        status=OrderStatus(record.status),
        updated_at=record.updated_at.date().isoformat(),
        estimated_delivery=(
            record.estimated_delivery_at.isoformat()
            if record.estimated_delivery_at is not None
            else None
        ),
    )


def cancellation_request_fingerprint(order_id: str) -> str:
    """Bind a retry key to one canonical cancellation request without storing raw input."""

    canonical_request = json.dumps(
        {"operation": CANCEL_ORDER_OPERATION, "order_id": order_id},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()


def _find_idempotency_record(
    session: Session,
    user_id: str,
    idempotency_key: str,
) -> IdempotencyRecord | None:
    return session.get(
        IdempotencyRecord,
        {
            "actor_id": user_id,
            "operation": CANCEL_ORDER_OPERATION,
            "idempotency_key": idempotency_key,
        },
    )


def _idempotency_result(
    record: IdempotencyRecord,
    fingerprint: str,
) -> OrderStatusData:
    if record.request_fingerprint != fingerprint:
        raise IdempotencyKeyConflict("Idempotency key conflicts with another request.")
    try:
        return OrderStatusData.model_validate(record.result_payload)
    except ValueError as error:
        raise OrderRepositoryUnavailable("Stored idempotency result is invalid.") from error
