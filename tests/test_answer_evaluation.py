from ai_support_agent.rag.answer_evaluation import AnswerEvaluationCase, evaluate_answer
from ai_support_agent.schemas import AnswerStatus, SupportResponse


REFUND_CASE = AnswerEvaluationCase(
    question="When will my refund arrive?",
    expected_status=AnswerStatus.ANSWERED,
    expected_source_ids=frozenset({"refund-policy-v1"}),
    required_phrases=("3-7 working days",),
    forbidden_phrases=("14 days",),
)


def test_answer_evaluation_accepts_a_grounded_response_despite_dash_variant() -> None:
    response = SupportResponse(
        status=AnswerStatus.ANSWERED,
        answer="Refund takes 3–7 working days.",
        alternative=None,
        recommendations=[],
        sources=["refund-policy-v1"],
    )

    result = evaluate_answer(REFUND_CASE, response)

    assert result.passed is True
    assert result.failures == ()


def test_answer_evaluation_reports_status_source_and_unsupported_fact_failures() -> None:
    response = SupportResponse(
        status=AnswerStatus.INSUFFICIENT_CONTEXT,
        answer="Refund takes 14 days.",
        alternative=None,
        recommendations=[],
        sources=["delivery-policy-v1"],
    )

    result = evaluate_answer(REFUND_CASE, response)

    assert result.passed is False
    assert len(result.failures) == 4
    assert any("Expected status" in failure for failure in result.failures)
    assert any("Expected sources" in failure for failure in result.failures)
    assert any("Missing required phrase" in failure for failure in result.failures)
    assert any("forbidden phrase" in failure for failure in result.failures)
