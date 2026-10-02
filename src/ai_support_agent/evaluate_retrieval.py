"""Manual RAG evaluation command that uses embeddings but never calls an LLM."""

import argparse

from ai_support_agent.exceptions import ConfigurationError, EmbeddingRequestError, VectorStoreError
from ai_support_agent.service import (
    RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
    RETRIEVAL_TOP_K,
)
from ai_support_agent.rag.evaluation import (
    DEFAULT_RETRIEVAL_CASES,
    DEFAULT_THRESHOLDS,
    EXTERNAL_REFERENCE_RETRIEVAL_CASES,
    RetrievalEvaluationResult,
    ThresholdEvaluationSummary,
    evaluate_thresholds,
)
from ai_support_agent.rag.runtime import (
    create_external_reference_gemini_vector_retriever,
    create_gemini_vector_retriever,
)


def format_result(result: RetrievalEvaluationResult) -> str:
    """Render one retrieval evaluation in a compact human-readable form."""

    source_scores = ", ".join(
        f"{chunk.document_id} ({chunk.score:.3f})"
        for chunk in result.retrieved_chunks
    ) or "ничего"
    metrics = (
        "ожидалось пусто"
        if result.recall_at_k is None
        else f"recall={result.recall_at_k:.2f}; precision={result.precision_at_k:.2f}"
    )


    status = "PASS" if result.passed else "FAIL"
    return "\n".join(
        (
            f"[{status}] {result.case.question}",
            f"  Найдено: {source_scores}",
            f"  {metrics}",
        )
    )


def format_summary(summary: ThresholdEvaluationSummary) -> str:
    """Render aggregate metrics for one threshold."""

    recall = (
        f"{summary.average_recall_at_k:.2f}"
        if summary.average_recall_at_k is not None
        else "n/a"
    )
    precision = (
        f"{summary.average_precision_at_k:.2f}"
        if summary.average_precision_at_k is not None
        else "n/a"
    )
    return (
        f"threshold={summary.threshold:.2f}; "
        f"passed={summary.passed_cases}/{len(summary.results)}; "
        f"avg_recall={recall}; avg_precision={precision}"
    )
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval without an LLM call.")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Similarity thresholds to compare.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Show retrieved source IDs and scores for every evaluation case.",
    )
    parser.add_argument(
        "--collection",
        choices=("internal", "external"),
        default="internal",
        help="Knowledge collection to index and evaluate.",
    )
    args = parser.parse_args()

    try:
        if args.collection == "external":
            retriever = create_external_reference_gemini_vector_retriever()
            cases = EXTERNAL_REFERENCE_RETRIEVAL_CASES
        else:
            retriever = create_gemini_vector_retriever()
            cases = DEFAULT_RETRIEVAL_CASES
        summaries = evaluate_thresholds(
            retriever,
            cases,
            args.thresholds,
            top_k=RETRIEVAL_TOP_K,
            max_chunks_per_document=RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
        )
    except ConfigurationError as error:
        print(f"Ошибка конфигурации: {error}")
        return
    except VectorStoreError:
        print("Векторная база недоступна. Проверьте контейнер Qdrant.")
        return
    except EmbeddingRequestError:
        print("Не удалось создать embeddings для проверки retrieval. Попробуйте позже.")
        return

    for summary in summaries:
        print(format_summary(summary))
        if args.details:
            for result in summary.results:
                print(format_result(result))


if __name__ == "__main__":
    main()
