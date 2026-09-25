from dataclasses import dataclass

from pydantic import ValidationError

from ai_support_agent.exceptions import InvalidModelResponseError
from ai_support_agent.llm_client import LlmCall, LlmClient
from ai_support_agent.rag.retriever import RetrievedChunk, Retriever
from ai_support_agent.schemas import SupportResponse

SYSTEM_PROMPT = """Ты — AI Support Agent.
Отвечай только на русском языке, кратко и по делу.
Используй только факты из переданного контекста.
Не выдумывай данные и честно сообщай, когда информации недостаточно.
Верни ответ строго в согласованном JSON-формате.
"""

SOURCE_TYPE_RULE = """Use source_type when evaluating evidence.
internal_policy confirms the store's own rules. external_reference is only external
reference material: do not promise the store's deadline, status, or action based on it alone.
If only external_reference is available for such a question, say that exact store information
is unavailable and set status to insufficient_context. You may put verified general facts from
external_reference in alternative, starting with the Russian label "Справочно:" and explicitly
stating that this is not a confirmation of the store's own rule."""

NO_RELEVANT_CONTEXT = (
    "Релевантные фрагменты базы знаний не найдены. "
    "Не используй внутренние знания для ответа."
)
RETRIEVAL_TOP_K = 3
RETRIEVAL_THRESHOLD = 0.7
RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT = 2
EXTERNAL_REFERENCE_RETRIEVAL_THRESHOLD = 0.75


@dataclass(frozen=True)
class AnswerResult:
    """A validated response together with non-user-facing usage metrics."""

    response: SupportResponse
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """Format evidence and source metadata for the LLM prompt."""

    if not chunks:
        return NO_RELEVANT_CONTEXT

    return "\n\n".join(
        f"Источник: {chunk.title} ({chunk.document_id})\n"
        f"Тип источника: {chunk.source_type}\n"
        f"Страница: {chunk.page_number or 'не применимо'}\n"
        f"Факт: {chunk.text}"
        for chunk in chunks
    )


def source_ids_from_chunks(chunks: list[RetrievedChunk]) -> list[str]:
    """Return unique provenance IDs in retrieval order, never trusting LLM output."""

    return list(dict.fromkeys(chunk.document_id for chunk in chunks))


def build_llm_call(
    user_question: str,
    retrieved_context: str = NO_RELEVANT_CONTEXT,
) -> LlmCall:
    """Собирает полный контекст одного вызова модели."""
    return LlmCall(
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n{SOURCE_TYPE_RULE}"},
            {"role": "user", "content": f"Контекст:\n{retrieved_context}"},
            {"role": "user", "content": user_question},
        ]
    )


def answer_question(
    user_question: str,
    client: LlmClient,
    retriever: Retriever,
) -> AnswerResult:
    """Выполняет учебный путь: контекст -> LLM -> валидация ответа."""
    chunks = retriever.retrieve(
        user_question,
        top_k=RETRIEVAL_TOP_K,
        threshold=RETRIEVAL_THRESHOLD,
        max_chunks_per_document=RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT,
    )
    call = build_llm_call(user_question, format_retrieved_context(chunks))
    result = client.complete(call)
    try:
        response = SupportResponse.model_validate_json(result.text)
    except ValidationError as error:
        raise InvalidModelResponseError(
            "LLM response does not match the SupportResponse contract."
        ) from error

    response = response.model_copy(update={"sources": source_ids_from_chunks(chunks)})

    return AnswerResult(
        response=response,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.total_tokens,
    )
