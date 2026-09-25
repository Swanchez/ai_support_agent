import pytest
from dataclasses import dataclass
from pydantic import ValidationError

from ai_support_agent.exceptions import InvalidModelResponseError
from ai_support_agent.llm_client import FakeLlmClient, LlmResult
from ai_support_agent.rag.knowledge_base import DEFAULT_KNOWLEDGE_BASE
from ai_support_agent.rag.models import SourceType
from ai_support_agent.rag.retriever import RetrievedChunk
from ai_support_agent.schemas import (
    AnswerStatus,
    support_response_json_schema,
    support_response_openai_text_format,
)
from ai_support_agent.service import (
    AnswerResult,
    NO_RELEVANT_CONTEXT,
    answer_question,
    build_llm_call,
    format_retrieved_context,
    source_ids_from_chunks,
)


@dataclass
class StubRetriever:
    results: list[RetrievedChunk]
    received_question: str | None = None

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        _ = top_k, threshold, max_chunks_per_document
        self.received_question = question
        return self.results


def test_format_retrieved_context_includes_evidence_and_source() -> None:
    chunk = DEFAULT_KNOWLEDGE_BASE[0]
    context = format_retrieved_context(
        [
            RetrievedChunk(
                text=chunk.text,
                score=0.8,
                document_id=chunk.document_id,
                title=chunk.title,
            )
        ]
    )

    assert "Источник: Правила возврата (refund-policy-v1)" in context
    assert "Деньги за возврат" in context


def test_format_retrieved_context_marks_an_empty_search() -> None:
    assert format_retrieved_context([]) == NO_RELEVANT_CONTEXT


def test_format_retrieved_context_keeps_source_type_and_pdf_page() -> None:
    context = format_retrieved_context(
        [
            RetrievedChunk(
                text="External fact",
                score=0.8,
                document_id="consumer-rights-v1",
                title="Consumer rights",
                page_number=2,
                source_type=SourceType.EXTERNAL_REFERENCE,
            )
        ]
    )

    assert "external_reference" in context
    assert "2" in context


def test_build_llm_call_contains_rules_context_and_question() -> None:
    context = "Источник: Правила возврата (refund-policy-v1)\nФакт: Срок возврата 3–7 дней."
    call = build_llm_call("Когда придут деньги за возврат?", context)

    assert call.temperature == 0.2
    assert call.max_output_tokens == 200
    assert [message["role"] for message in call.messages] == ["system", "user", "user"]
    assert "не выдумывай" in call.messages[0]["content"].lower()
    assert context in call.messages[1]["content"]
    assert "external_reference" in call.messages[0]["content"]
    assert "alternative" in call.messages[0]["content"]
    assert call.messages[2]["content"] == "Когда придут деньги за возврат?"


def test_answer_is_validated_against_our_contract() -> None:
    retriever = StubRetriever([])
    response = answer_question("Когда придут деньги за возврат?", FakeLlmClient(), retriever)

    assert isinstance(response, AnswerResult)
    assert response.response.status is AnswerStatus.INSUFFICIENT_CONTEXT
    assert response.response.sources == []
    assert response.model == "fake"
    assert response.total_tokens == 0
    assert retriever.received_question == "Когда придут деньги за возврат?"


def test_invalid_model_response_is_wrapped_in_application_error() -> None:
    client = FakeLlmClient(raw_response='{"status": "done"}')

    with pytest.raises(InvalidModelResponseError) as error_info:
        answer_question("Когда придут деньги за возврат?", client, StubRetriever([]))

    assert isinstance(error_info.value.__cause__, ValidationError)


def test_fake_client_returns_a_result_with_usage_metrics() -> None:
    result = FakeLlmClient().complete(build_llm_call("Когда придут деньги за возврат?"))

    assert isinstance(result, LlmResult)
    assert result.model == "fake"
    assert result.total_tokens == 0


def test_support_response_schema_describes_our_contract() -> None:
    schema = support_response_json_schema()

    assert schema["type"] == "object"
    assert set(schema["required"]) == {
        "status",
        "answer",
        "alternative",
        "recommendations",
        "sources",
    }
    assert schema["properties"]["answer"]["minLength"] == 1
    assert schema["additionalProperties"] is False


def test_openai_text_format_uses_our_strict_schema() -> None:
    text_format = support_response_openai_text_format()

    assert text_format["format"]["type"] == "json_schema"
    assert text_format["format"]["name"] == "support_response"
    assert text_format["format"]["strict"] is True
    assert text_format["format"]["schema"] == support_response_json_schema()


def test_answer_sources_come_from_retrieval_not_llm_json() -> None:
    retrieved_chunks = [
        RetrievedChunk("First fact", 0.9, "document-a", "First source"),
        RetrievedChunk("Second fact", 0.8, "document-b", "Second source"),
        RetrievedChunk("Duplicate", 0.7, "document-a", "First source"),
    ]

    response = answer_question(
        "User question",
        FakeLlmClient(),
        StubRetriever(retrieved_chunks),
    )

    assert response.response.sources == ["document-a", "document-b"]


def test_source_ids_from_chunks_returns_empty_list_for_empty_retrieval() -> None:
    assert source_ids_from_chunks([]) == []
