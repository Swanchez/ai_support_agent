from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ai_support_agent.rag.embeddings import Embedding
from ai_support_agent.rag.index_cache import (
    IndexMetadata,
    knowledge_base_fingerprint,
    load_index,
    load_or_create_index,
    save_index,
)
from ai_support_agent.rag.indexing import embedding_input
from ai_support_agent.rag.knowledge_base import KnowledgeChunk
from ai_support_agent.rag.vector_store import IndexedChunk, InMemoryVectorStore


@dataclass
class FakeEmbeddingClient:
    calls: list[str] = field(default_factory=list)

    def embed(self, text: str) -> Embedding:
        self.calls.append(text)
        return (1.0, 0.0)


def metadata_for(chunks: tuple[KnowledgeChunk, ...]) -> IndexMetadata:
    return IndexMetadata(
        embedding_model="test-embedding-model",
        dimensions=2,
        knowledge_base_fingerprint=knowledge_base_fingerprint(chunks),
    )


TEST_CACHE_PATH = Path(__file__).parent / "_rag_index_cache_test.json"


@pytest.fixture
def cache_path() -> Path:
    """Use one disposable project-local file when temporary directories are restricted."""

    TEST_CACHE_PATH.unlink(missing_ok=True)
    try:
        yield TEST_CACHE_PATH
    finally:
        TEST_CACHE_PATH.unlink(missing_ok=True)


def test_knowledge_base_fingerprint_changes_when_searchable_text_changes() -> None:
    original = (KnowledgeChunk("refund-v1", "Возврат", "Срок 3–7 дней"),)
    changed = (KnowledgeChunk("refund-v1", "Возврат", "Срок 5–10 дней"),)

    assert knowledge_base_fingerprint(original) != knowledge_base_fingerprint(changed)


def test_save_then_load_returns_a_compatible_index(cache_path: Path) -> None:
    chunks = (KnowledgeChunk("refund-v1", "Возврат", "Срок 3–7 дней"),)
    metadata = metadata_for(chunks)
    index = InMemoryVectorStore((IndexedChunk(chunks[0], (0.6, 0.8)),))

    save_index(cache_path, index, metadata)

    assert load_index(cache_path, metadata) == index


def test_load_index_rejects_metadata_for_a_changed_knowledge_base(cache_path: Path) -> None:
    original = (KnowledgeChunk("refund-v1", "Возврат", "Срок 3–7 дней"),)
    changed = (KnowledgeChunk("refund-v1", "Возврат", "Срок 5–10 дней"),)
    save_index(
        cache_path,
        InMemoryVectorStore((IndexedChunk(original[0], (0.6, 0.8)),)),
        metadata_for(original),
    )

    assert load_index(cache_path, metadata_for(changed)) is None


def test_load_or_create_index_reuses_saved_vectors_without_embedding_again(cache_path: Path) -> None:
    chunks = (KnowledgeChunk("refund-v1", "Возврат", "Срок 3–7 дней"),)
    metadata = metadata_for(chunks)
    first_client = FakeEmbeddingClient()

    load_or_create_index(cache_path, chunks, first_client, metadata)

    second_client = FakeEmbeddingClient()
    loaded_index = load_or_create_index(cache_path, chunks, second_client, metadata)

    assert first_client.calls == [embedding_input(chunks[0])]
    assert second_client.calls == []
    assert loaded_index.indexed_chunks[0].chunk == chunks[0]
