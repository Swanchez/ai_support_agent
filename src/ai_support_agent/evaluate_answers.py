"""Manual live evaluation of complete RAG answers; this command calls configured APIs."""

from ai_support_agent.exceptions import (
    ConfigurationError,
    EmbeddingRequestError,
    InvalidModelResponseError,
    LlmRequestError,
)
from ai_support_agent.factory import create_llm_client
from ai_support_agent.rag.answer_evaluation import (
    DEFAULT_ANSWER_EVALUATION_CASES,
    AnswerEvaluationResult,
    evaluate_answer,
)
from ai_support_agent.rag.fallback_retriever import FallbackRetriever
from ai_support_agent.rag.runtime import (
    create_external_reference_gemini_vector_retriever,
    create_gemini_vector_retriever,
)
from ai_support_agent.service import EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD, answer_question


def create_support_retriever() -> FallbackRetriever:
    """Assemble the same source-aware retrieval policy used by the console application."""

    return FallbackRetriever(
        primary=create_gemini_vector_retriever(),
        fallback_factory=create_external_reference_gemini_vector_retriever,
        fallback_threshold=EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD,
    )


def format_result(result: AnswerEvaluationResult) -> str:
    """Show automatic checks and enough answer text for a focused human review."""

    response = result.response
    status = "PASS" if result.passed else "FAIL"
    lines = [
        f"[{status}] {result.case.question}",
        f"  status={response.status}; sources={', '.join(response.sources) or 'nothing'}",
        f"  answer={response.answer}",
    ]
    if response.alternative:
        lines.append(f"  reference={response.alternative}")
    lines.extend(f"  failure={failure}" for failure in result.failures)
    return "\n".join(lines)


def main() -> None:
    """Call the application once per evaluation case, then check the returned responses."""

    try:
        client = create_llm_client()
        retriever = create_support_retriever()
        for case in DEFAULT_ANSWER_EVALUATION_CASES:
            answer_result = answer_question(case.question, client, retriever)
            print(format_result(evaluate_answer(case, answer_result.response)))
    except ConfigurationError as error:
        print(f"Configuration error: {error}")
    except (LlmRequestError, EmbeddingRequestError):
        print("Could not call the configured model or embedding service.")
    except InvalidModelResponseError:
        print("The model returned an invalid support response.")


if __name__ == "__main__":
    main()
