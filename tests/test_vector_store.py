import pytest

from ai_support_agent.rag.knowledge_base import KnowledgeChunk
from ai_support_agent.rag.vector_store import (
    IndexedChunk,
    InMemoryVectorStore,
    cosine_similarity,
)


def test_cosine_similarity_measures_vector_direction() -> None:
    assert cosine_similarity((1.0, 0.0), (3.0, 0.0)) == pytest.approx(1.0)
    assert cosine_similarity((1.0, 0.0), (0.0, 1.0)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("left", "right"),
    [((1.0,), (1.0, 0.0)), ((0.0, 0.0), (1.0, 0.0))],
)
def test_cosine_similarity_rejects_incomparable_vectors(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError):
        cosine_similarity(left, right)


def test_vector_store_returns_metadata_in_similarity_order() -> None:
    refund_chunk = KnowledgeChunk("refund-v1", "Возврат", "Возврат денег")
    delivery_chunk = KnowledgeChunk("delivery-v1", "Доставка", "Срок доставки")
    store = InMemoryVectorStore(
        (
            IndexedChunk(refund_chunk, (1.0, 0.0)),
            IndexedChunk(delivery_chunk, (0.6, 0.8)),
        )
    )

    results = store.search((1.0, 0.0), top_k=1, threshold=0.5)

    assert [result.document_id for result in results] == ["refund-v1"]
    assert results[0].title == "Возврат"
    assert results[0].score == pytest.approx(1.0)


def test_vector_store_filters_distant_chunks() -> None:
    chunk = KnowledgeChunk("delivery-v1", "Доставка", "Срок доставки")
    store = InMemoryVectorStore((IndexedChunk(chunk, (0.0, 1.0)),))

    assert store.search((1.0, 0.0), threshold=0.2) == []


def test_vector_store_limits_chunks_per_document_but_keeps_other_sources() -> None:
    first = KnowledgeChunk("refund-v1", "Refund", "First", "refund-v1#1")
    second = KnowledgeChunk("refund-v1", "Refund", "Second", "refund-v1#2")
    delivery = KnowledgeChunk("delivery-v1", "Delivery", "Delivery", "delivery-v1#1")
    store = InMemoryVectorStore(
        (
            IndexedChunk(first, (1.0, 0.0)),
            IndexedChunk(second, (0.95, 0.05)),
            IndexedChunk(delivery, (0.8, 0.6)),
        )
    )

    results = store.search(
        (1.0, 0.0),
        top_k=2,
        threshold=0.0,
        max_chunks_per_document=1,
    )

    assert [result.document_id for result in results] == ["refund-v1", "delivery-v1"]
