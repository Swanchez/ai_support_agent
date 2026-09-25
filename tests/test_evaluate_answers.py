from ai_support_agent.evaluate_answers import format_result
from ai_support_agent.rag.answer_evaluation import AnswerEvaluationCase, evaluate_answer
from ai_support_agent.schemas import AnswerStatus, SupportResponse


def test_format_result_shows_automatic_failure_and_answer_for_human_review() -> None:
    case = AnswerEvaluationCase(
        question="Refund question",
        expected_status=AnswerStatus.ANSWERED,
        expected_source_ids=frozenset({"refund-v1"}),
        required_phrases=("3-7 days",),
    )
    response = SupportResponse(
        status=AnswerStatus.ANSWERED,
        answer="Refund takes 14 days.",
        alternative=None,
        recommendations=[],
        sources=["refund-v1"],
    )

    rendered = format_result(evaluate_answer(case, response))

    assert rendered.startswith("[FAIL] Refund question")
    assert "answer=Refund takes 14 days." in rendered
    assert "Missing required phrase" in rendered
