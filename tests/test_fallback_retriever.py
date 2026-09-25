from dataclasses import dataclass, field

from ai_support_agent.rag.fallback_retriever import FallbackRetriever
from ai_support_agent.rag.retriever import RetrievedChunk


@dataclass
class SpyRetriever:
    results: list[RetrievedChunk]
    calls: list[tuple[str, int, float, int | None]] = field(default_factory=list)

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((question, top_k, threshold, max_chunks_per_document))
        return self.results


INTERNAL_RESULT = RetrievedChunk(
    text="Store policy",
    score=0.9,
    document_id="refund-policy-v1",
    title="Refund policy",
)
EXTERNAL_RESULT = RetrievedChunk(
    text="External reference",
    score=0.8,
    document_id="consumer-rights-v1",
    title="Consumer rights",
)


def test_fallback_retriever_keeps_primary_result_and_does_not_search_fallback() -> None:
    primary = SpyRetriever([INTERNAL_RESULT])
    fallback = SpyRetriever([EXTERNAL_RESULT])
    fallback_factory_calls = 0

    def create_fallback() -> SpyRetriever:
        nonlocal fallback_factory_calls
        fallback_factory_calls += 1
        return fallback

    retriever = FallbackRetriever(primary, create_fallback, fallback_threshold=0.75)

    results = retriever.retrieve("When is my refund?", top_k=3, threshold=0.7)

    assert results == [INTERNAL_RESULT]
    assert len(primary.calls) == 1
    assert fallback.calls == []
    assert fallback_factory_calls == 0


def test_fallback_retriever_searches_external_collection_only_when_primary_is_empty() -> None:
    primary = SpyRetriever([])
    fallback = SpyRetriever([EXTERNAL_RESULT])
    retriever = FallbackRetriever(primary, lambda: fallback, fallback_threshold=0.75)

    results = retriever.retrieve(
        "What are consumer rights?",
        top_k=2,
        threshold=0.6,
        max_chunks_per_document=1,
    )

    assert results == [EXTERNAL_RESULT]
    assert primary.calls == [("What are consumer rights?", 2, 0.6, 1)]
    assert fallback.calls == [("What are consumer rights?", 2, 0.75, 1)]
