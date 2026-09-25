from dataclasses import dataclass

from ai_support_agent.rag.evaluation import (
    EXTERNAL_REFERENCE_RETRIEVAL_CASES,
    RetrievalCase,
    evaluate_retriever,
    evaluate_thresholds,
)
from ai_support_agent.rag.retriever import RetrievedChunk


@dataclass
class StubRetriever:
    results_by_question: dict[str, list[RetrievedChunk]]

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        _ = top_k, threshold, max_chunks_per_document
        return self.results_by_question[question]


def test_evaluate_retriever_calculates_recall_and_precision() -> None:
    case = RetrievalCase("refund question", frozenset({"refund-v1"}))
    retriever = StubRetriever(
        {
            "refund question": [
                RetrievedChunk("refund", 0.9, "refund-v1", "Refund"),
                RetrievedChunk("delivery", 0.7, "delivery-v1", "Delivery"),
            ]
        }
    )

    [result] = evaluate_retriever(retriever, [case])

    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 0.5
    assert result.passed is True


def test_evaluate_retriever_treats_empty_retrieval_as_correct_for_unknown_question() -> None:
    case = RetrievalCase("unknown question", frozenset())
    retriever = StubRetriever({"unknown question": []})

    [result] = evaluate_retriever(retriever, [case])

    assert result.recall_at_k is None
    assert result.precision_at_k is None
    assert result.passed is True


def test_evaluate_retriever_marks_irrelevant_result_for_unknown_question_as_failed() -> None:
    case = RetrievalCase("unknown question", frozenset())
    retriever = StubRetriever(
        {"unknown question": [RetrievedChunk("delivery", 0.4, "delivery-v1", "Delivery")]}
    )

    [result] = evaluate_retriever(retriever, [case])

    assert result.passed is False


def test_evaluate_thresholds_reuses_one_retrieval_per_case() -> None:
    case = RetrievalCase("refund question", frozenset({"refund-v1"}))

    @dataclass
    class CountingRetriever:
        calls: int = 0

        def retrieve(
            self,
            question: str,
            *,
            top_k: int,
            threshold: float,
            max_chunks_per_document: int | None = None,
        ) -> list[RetrievedChunk]:
            _ = question, top_k, max_chunks_per_document
            assert threshold == 0.0
            self.calls += 1
            return [RetrievedChunk("refund", 0.5, "refund-v1", "Refund")]

    retriever = CountingRetriever()

    summaries = evaluate_thresholds(retriever, [case], thresholds=[0.4, 0.6])

    assert retriever.calls == 1
    assert summaries[0].passed_cases == 1
    assert summaries[1].passed_cases == 0


def test_external_reference_cases_cover_both_documents_and_a_negative_query() -> None:
    expected_ids = {
        document_id
        for case in EXTERNAL_REFERENCE_RETRIEVAL_CASES
        for document_id in case.expected_document_ids
    }

    assert expected_ids == {
        "consumer-remote-sales-v1",
        "consumer-exchange-return-v1",
    }
    assert any(not case.expected_document_ids for case in EXTERNAL_REFERENCE_RETRIEVAL_CASES)
