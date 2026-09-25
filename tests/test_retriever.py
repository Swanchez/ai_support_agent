import pytest

from ai_support_agent.rag.knowledge_base import DEFAULT_KNOWLEDGE_BASE, KnowledgeChunk
from ai_support_agent.rag.retriever import KeywordRetriever, tokenize


def test_tokenize_normalizes_russian_words() -> None:
    assert tokenize("Деньги, деньги за ВОЗВРАТ!") == {"деньги", "за", "возврат"}


def test_keyword_retriever_returns_the_refund_chunk_with_metadata() -> None:
    retriever = KeywordRetriever(DEFAULT_KNOWLEDGE_BASE)

    results = retriever.retrieve(
        "Когда вернут деньги за возврат?",
        top_k=1,
        threshold=0.2,
    )

    assert len(results) == 1
    assert results[0].document_id == "refund-policy-v1"
    assert results[0].title == "Правила возврата"
    assert results[0].score > 0


def test_keyword_retriever_filters_weak_matches() -> None:
    retriever = KeywordRetriever(DEFAULT_KNOWLEDGE_BASE)

    results = retriever.retrieve(
        "Как изменить адрес доставки?",
        top_k=3,
        threshold=0.5,
    )

    assert results == []


def test_keyword_retriever_orders_results_and_limits_top_k() -> None:
    chunks = (
        KnowledgeChunk("refund-short", "Возврат", "Возврат возможен."),
        KnowledgeChunk("refund-full", "Возврат", "Возврат денег возможен."),
    )
    retriever = KeywordRetriever(chunks)

    results = retriever.retrieve("Возврат денег", top_k=1, threshold=0)

    assert [result.document_id for result in results] == ["refund-full"]


@pytest.mark.parametrize(
    ("top_k", "threshold"),
    [(0, 0.2), (1, -0.1), (1, 1.1)],
)
def test_keyword_retriever_rejects_invalid_search_limits(
    top_k: int,
    threshold: float,
) -> None:
    retriever = KeywordRetriever(DEFAULT_KNOWLEDGE_BASE)

    with pytest.raises(ValueError):
        retriever.retrieve("возврат", top_k=top_k, threshold=threshold)
