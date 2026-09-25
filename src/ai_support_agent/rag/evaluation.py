"""Small, explicit evaluation helpers for measuring retrieval quality."""

from dataclasses import dataclass
from typing import Iterable

from ai_support_agent.rag.retriever import RetrievedChunk, Retriever


@dataclass(frozen=True)
class RetrievalCase:
    """One question and the source IDs that a good retriever should find."""

    question: str
    expected_document_ids: frozenset[str]


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    """Metrics for one retrieval case at one fixed value of top_k."""

    case: RetrievalCase
    retrieved_chunks: list[RetrievedChunk]
    recall_at_k: float | None
    precision_at_k: float | None
    passed: bool


@dataclass(frozen=True)
class ThresholdEvaluationSummary:
    """Aggregate retrieval quality for one threshold across a fixed case set."""

    threshold: float
    results: list[RetrievalEvaluationResult]
    passed_cases: int
    average_recall_at_k: float | None
    average_precision_at_k: float | None


DEFAULT_RETRIEVAL_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        "Когда придут деньги за возврат?",
        frozenset({"refund-policy-v1"}),
    ),
    RetrievalCase(
        "Через сколько зачислят средства после отправки товара обратно?",
        frozenset({"refund-policy-v1"}),
    ),
    RetrievalCase(
        "Как долго везут заказ?",
        frozenset({"delivery-policy-v1"}),
    ),
    RetrievalCase(
        "Как поменять адрес доставки?",
        frozenset(),
    ),
)

EXTERNAL_REFERENCE_RETRIEVAL_CASES: tuple[RetrievalCase, ...] = (
    RetrievalCase(
        "В какой срок после получения можно отказаться от товара, купленного через интернет?",
        frozenset({"consumer-remote-sales-v1"}),
    ),
    RetrievalCase(
        "Когда продавец возвращает деньги при отказе от товара, заказанного дистанционно?",
        frozenset({"consumer-remote-sales-v1"}),
    ),
    RetrievalCase(
        "В течение какого срока можно обменять непродовольственный товар надлежащего качества?",
        frozenset({"consumer-exchange-return-v1"}),
    ),
    RetrievalCase(
        "Как оформить банковский кредит?",
        frozenset(),
    ),
)
DEFAULT_THRESHOLDS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)


def evaluate_retriever(
    retriever: Retriever,
    cases: Iterable[RetrievalCase],
    *,
    top_k: int = 3,
    threshold: float = 0.2,
    max_chunks_per_document: int | None = None,
) -> list[RetrievalEvaluationResult]:
    """Evaluate source selection without calling an LLM or judging answer wording."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    results: list[RetrievalEvaluationResult] = []
    for case in cases:
        retrieved_chunks = retriever.retrieve(
            case.question,
            top_k=top_k,
            threshold=threshold,
            max_chunks_per_document=max_chunks_per_document,
        )
        results.append(_evaluate_case(case, retrieved_chunks))

    return results


def evaluate_thresholds(
    retriever: Retriever,
    cases: Iterable[RetrievalCase],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    *,
    top_k: int = 3,
    max_chunks_per_document: int | None = None,
) -> list[ThresholdEvaluationSummary]:
    """Compare thresholds while embedding each evaluation question only once."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    threshold_values = tuple(thresholds)
    if any(not 0 <= threshold <= 1 for threshold in threshold_values):
        raise ValueError("thresholds must be between 0 and 1.")

    cases_tuple = tuple(cases)
    candidates_by_case = [
        (
            case,
            retriever.retrieve(
                case.question,
                top_k=top_k,
                threshold=0.0,
                max_chunks_per_document=max_chunks_per_document,
            ),
        )
        for case in cases_tuple
    ]

    summaries: list[ThresholdEvaluationSummary] = []
    for threshold in threshold_values:
        results = [
            _evaluate_case(
                case,
                [chunk for chunk in candidates if chunk.score >= threshold],
            )
            for case, candidates in candidates_by_case
        ]
        known_source_results = [result for result in results if result.recall_at_k is not None]
        summaries.append(
            ThresholdEvaluationSummary(
                threshold=threshold,
                results=results,
                passed_cases=sum(result.passed for result in results),
                average_recall_at_k=(
                    sum(result.recall_at_k for result in known_source_results)
                    / len(known_source_results)
                    if known_source_results
                    else None
                ),
                average_precision_at_k=(
                    sum(result.precision_at_k for result in known_source_results)
                    / len(known_source_results)
                    if known_source_results
                    else None
                ),
            )
        )

    return summaries


def _evaluate_case(
    case: RetrievalCase,
    retrieved_chunks: list[RetrievedChunk],
) -> RetrievalEvaluationResult:
    """Calculate per-case metrics from already retrieved chunks."""

    expected_ids = case.expected_document_ids

    if not expected_ids:
        return RetrievalEvaluationResult(
            case=case,
            retrieved_chunks=retrieved_chunks,
            recall_at_k=None,
            precision_at_k=None,
            passed=not retrieved_chunks,
        )

    retrieved_ids = [chunk.document_id for chunk in retrieved_chunks]
    matched_expected_ids = expected_ids.intersection(retrieved_ids)
    recall_at_k = len(matched_expected_ids) / len(expected_ids)
    precision_at_k = (
        sum(chunk.document_id in expected_ids for chunk in retrieved_chunks)
        / len(retrieved_chunks)
        if retrieved_chunks
        else 0.0
    )
    return RetrievalEvaluationResult(
        case=case,
        retrieved_chunks=retrieved_chunks,
        recall_at_k=recall_at_k,
        precision_at_k=precision_at_k,
        passed=recall_at_k == 1.0,
    )
