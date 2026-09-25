from ai_support_agent.agents.core import (
    AgentAction,
    AgentObservation,
    AgentRunResult,
    AgentState,
    AgentToolResult,
)
from ai_support_agent.agents.evaluation import (
    AgentEvaluationCase,
    ExpectedAgentAction,
    evaluate_agent_run,
)
from ai_support_agent.schemas import AnswerStatus, SupportResponse


def run_result(*observations: AgentObservation, step_limit_reached: bool = False) -> AgentRunResult:
    return AgentRunResult(
        response=SupportResponse(
            status=AnswerStatus.ANSWERED,
            answer="Ответ на основе источников.",
            alternative=None,
            recommendations=[],
            sources=[
                source_id
                for observation in observations
                for source_id in observation.result.source_ids
            ],
        ),
        state=AgentState(
            original_question="Какие правила применимы?",
            observations=list(observations),
            step_count=len(observations),
        ),
        step_limit_reached=step_limit_reached,
    )


def test_evaluator_accepts_required_actions_arguments_and_sources() -> None:
    result = run_result(
        AgentObservation(
            AgentAction("get_order_status", {"order_id": "ORD-1001"}),
            AgentToolResult({"ok": True}, ("get_order_status",)),
        ),
        AgentObservation(
            AgentAction("search_knowledge_base", {"query": "правила возврата"}),
            AgentToolResult({"ok": True}, ("refund-policy-v1",)),
        ),
    )
    case = AgentEvaluationCase(
        question="Какие правила возврата применимы к ORD-1001?",
        required_actions=(
            ExpectedAgentAction("get_order_status", {"order_id": "ORD-1001"}),
            ExpectedAgentAction("search_knowledge_base"),
        ),
        forbidden_action_names=frozenset({"cancel_order"}),
        expected_status=AnswerStatus.ANSWERED,
        expected_source_ids=frozenset({"get_order_status", "refund-policy-v1"}),
        max_steps=3,
        allow_step_limit_reached=False,
    )

    evaluation = evaluate_agent_run(case, result)

    assert evaluation.passed
    assert evaluation.failures == ()


def test_evaluator_reports_missing_unsafe_and_unbounded_behaviour() -> None:
    result = run_result(
        AgentObservation(
            AgentAction("cancel_order", {"order_id": "ORD-1003"}),
            AgentToolResult({"ok": True}, ("cancel_order",)),
        ),
        step_limit_reached=True,
    )
    case = AgentEvaluationCase(
        question="Где заказ ORD-1001?",
        required_actions=(ExpectedAgentAction("get_order_status", {"order_id": "ORD-1001"}),),
        forbidden_action_names=frozenset({"cancel_order"}),
        expected_source_ids=frozenset({"get_order_status"}),
        max_steps=0,
        allow_step_limit_reached=False,
    )

    evaluation = evaluate_agent_run(case, result)

    assert not evaluation.passed
    assert any("get_order_status" in failure for failure in evaluation.failures)
    assert any("cancel_order" in failure for failure in evaluation.failures)
    assert any("Expected sources" in failure for failure in evaluation.failures)
    assert any("Expected at most 0" in failure for failure in evaluation.failures)
    assert any("step limit" in failure for failure in evaluation.failures)


def test_evaluator_reports_actions_outside_the_case_allowlist() -> None:
    result = run_result(
        AgentObservation(
            AgentAction("get_order_status", {"order_id": "ORD-1001"}),
            AgentToolResult({"ok": True}, ("get_order_status",)),
        ),
        AgentObservation(
            AgentAction("search_knowledge_base", {"query": "лишний поиск"}),
            AgentToolResult({"ok": True}, ("refund-policy-v1",)),
        ),
    )
    case = AgentEvaluationCase(
        question="Где заказ ORD-1001?",
        allowed_action_names=frozenset({"get_order_status"}),
    )

    evaluation = evaluate_agent_run(case, result)

    assert not evaluation.passed
    assert evaluation.failures == (
        "Unexpected action was executed: 'search_knowledge_base'.",
    )


def test_evaluator_accepts_one_honest_not_found_observation_without_retries() -> None:
    result = AgentRunResult(
        response=SupportResponse(
            status=AnswerStatus.INSUFFICIENT_CONTEXT,
            answer="Заказ ORD-9999 не найден.",
            alternative=None,
            recommendations=[],
            sources=["get_order_status"],
        ),
        state=AgentState(
            original_question="Где заказ ORD-9999?",
            observations=[
                AgentObservation(
                    AgentAction("get_order_status", {"order_id": "ORD-9999"}),
                    AgentToolResult({"ok": False, "code": "order_not_found"}, ("get_order_status",)),
                )
            ],
            step_count=1,
        ),
        step_limit_reached=False,
    )
    case = AgentEvaluationCase(
        question="Где заказ ORD-9999?",
        required_actions=(ExpectedAgentAction("get_order_status", {"order_id": "ORD-9999"}),),
        allowed_action_names=frozenset({"get_order_status"}),
        expected_status=AnswerStatus.INSUFFICIENT_CONTEXT,
        expected_source_ids=frozenset({"get_order_status"}),
        max_steps=1,
        allow_step_limit_reached=False,
    )

    evaluation = evaluate_agent_run(case, result)

    assert evaluation.passed


def test_evaluator_accepts_one_temporary_error_without_an_agent_retry() -> None:
    result = AgentRunResult(
        response=SupportResponse(
            status=AnswerStatus.INSUFFICIENT_CONTEXT,
            answer="Сервис заказов временно недоступен. Попробуйте позже.",
            alternative=None,
            recommendations=["Повторите попытку позже."],
            sources=["get_order_status"],
        ),
        state=AgentState(
            original_question="Где заказ ORD-1001?",
            observations=[
                AgentObservation(
                    AgentAction("get_order_status", {"order_id": "ORD-1001"}),
                    AgentToolResult(
                        {"ok": False, "code": "order_service_unavailable"},
                        ("get_order_status",),
                    ),
                )
            ],
            step_count=1,
        ),
        step_limit_reached=False,
    )
    case = AgentEvaluationCase(
        question="Где заказ ORD-1001?",
        required_actions=(ExpectedAgentAction("get_order_status", {"order_id": "ORD-1001"}),),
        allowed_action_names=frozenset({"get_order_status"}),
        expected_status=AnswerStatus.INSUFFICIENT_CONTEXT,
        expected_source_ids=frozenset({"get_order_status"}),
        max_steps=1,
        allow_step_limit_reached=False,
    )

    evaluation = evaluate_agent_run(case, result)

    assert evaluation.passed


def test_evaluator_accepts_any_status_from_an_explicit_allowed_set() -> None:
    result = run_result()
    result.response.status = AnswerStatus.CLARIFICATION_NEEDED
    case = AgentEvaluationCase(
        question="Какие правила возврата применимы к заказу?",
        allowed_statuses=frozenset(
            {AnswerStatus.ANSWERED, AnswerStatus.CLARIFICATION_NEEDED}
        ),
    )

    evaluation = evaluate_agent_run(case, result)

    assert evaluation.passed


def test_evaluator_rejects_two_competing_status_expectations() -> None:
    try:
        AgentEvaluationCase(
            question="test",
            expected_status=AnswerStatus.ANSWERED,
            allowed_statuses=frozenset({AnswerStatus.ANSWERED}),
        )
    except ValueError as error:
        assert "expected_status or allowed_statuses" in str(error)
    else:
        raise AssertionError("Competing status expectations should be rejected.")
