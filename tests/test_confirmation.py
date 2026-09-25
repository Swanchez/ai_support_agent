from ai_support_agent.tools.confirmation import InMemoryPendingActionStore
from ai_support_agent.tools.audit import AuditEventType, InMemoryAuditSink


def test_confirmed_action_is_bound_to_user_and_consumed_once() -> None:
    store = InMemoryPendingActionStore()
    pending = store.create(
        user_id="user-1",
        tool_name="cancel_order",
        arguments={"order_id": "ORD-1001"},
    )

    assert store.approve_for_user(
        confirmation_id=pending.confirmation_id,
        user_id="user-2",
    ) is None

    approved = store.approve_for_user(
        confirmation_id=pending.confirmation_id,
        user_id="user-1",
    )

    assert approved == pending
    assert store.approve_for_user(
        confirmation_id=pending.confirmation_id,
        user_id="user-1",
    ) is None


def test_store_keeps_its_own_copy_of_action_arguments() -> None:
    store = InMemoryPendingActionStore()
    arguments = {"order_id": "ORD-1001"}

    pending = store.create(
        user_id="user-1",
        tool_name="cancel_order",
        arguments=arguments,
    )
    arguments["order_id"] = "ORD-1002"

    assert pending.arguments == {"order_id": "ORD-1001"}


def test_store_audits_confirmation_lifecycle_without_arguments() -> None:
    audit_sink = InMemoryAuditSink()
    store = InMemoryPendingActionStore(audit_sink=audit_sink)
    approved = store.create(
        user_id="user-1",
        tool_name="cancel_order",
        arguments={"order_id": "ORD-1001"},
    )
    rejected = store.create(
        user_id="user-2",
        tool_name="cancel_order",
        arguments={"order_id": "ORD-1002"},
    )

    assert store.approve_for_user(
        confirmation_id=approved.confirmation_id,
        user_id="user-1",
    ) == approved
    assert store.discard_for_user(
        confirmation_id=rejected.confirmation_id,
        user_id="user-2",
    ) is True

    assert [event.event_type for event in audit_sink.events] == [
        AuditEventType.CONFIRMATION,
        AuditEventType.CONFIRMATION,
        AuditEventType.CONFIRMATION,
        AuditEventType.CONFIRMATION,
    ]
    assert [event.outcome for event in audit_sink.events] == [
        "proposal_created",
        "proposal_created",
        "confirmation_approved",
        "confirmation_rejected",
    ]
    assert all(not hasattr(event, "arguments") for event in audit_sink.events)
