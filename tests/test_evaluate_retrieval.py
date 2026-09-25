from ai_support_agent.evaluate_retrieval import format_result, format_summary
from ai_support_agent.rag.evaluation import (
    RetrievalCase,
    RetrievalEvaluationResult,
    ThresholdEvaluationSummary,
)
from ai_support_agent.rag.retriever import RetrievedChunk


def test_format_result_shows_source_score_and_metrics() -> None:
    result = RetrievalEvaluationResult(
        case=RetrievalCase("Refund question", frozenset({"refund-v1"})),
        retrieved_chunks=[RetrievedChunk("fact", 0.875, "refund-v1", "Refund")],
        recall_at_k=1.0,
        precision_at_k=1.0,
        passed=True,
    )

    text = format_result(result)

    assert "[PASS] Refund question" in text
    assert "refund-v1 (0.875)" in text
    assert "recall=1.00; precision=1.00" in text


def test_format_summary_shows_aggregate_threshold_metrics() -> None:
    summary = ThresholdEvaluationSummary(
        threshold=0.6,
        results=[],
        passed_cases=3,
        average_recall_at_k=0.75,
        average_precision_at_k=0.5,
    )

    assert format_summary(summary) == (
        "threshold=0.60; passed=3/0; avg_recall=0.75; avg_precision=0.50"
    )
