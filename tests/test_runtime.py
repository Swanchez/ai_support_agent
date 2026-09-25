from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ai_support_agent.rag.embeddings import Embedding
from ai_support_agent.rag.indexing import embedding_input
from ai_support_agent.rag.knowledge_base import (
    DEFAULT_KNOWLEDGE_BASE,
    load_external_reference_knowledge_base,
)
from ai_support_agent.rag.runtime import (
    create_external_reference_gemini_vector_retriever,
    create_gemini_vector_retriever,
)


@dataclass
class FakeEmbeddingClient:
    calls: list[str] = field(default_factory=list)

    def embed(self, text: str) -> Embedding:
        self.calls.append(text)
        return (1.0,) + (0.0,) * 767


TEST_INDEX_PATH = Path(__file__).parent / "_rag_runtime_test.json"


@pytest.fixture
def index_path() -> Path:
    TEST_INDEX_PATH.unlink(missing_ok=True)
    try:
        yield TEST_INDEX_PATH
    finally:
        TEST_INDEX_PATH.unlink(missing_ok=True)


def test_runtime_creates_retriever_and_indexes_the_default_knowledge_base(
    index_path: Path,
) -> None:
    embedding_client = FakeEmbeddingClient()
    retriever = create_gemini_vector_retriever(
        {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_EMBEDDING_MODEL": "test-embedding-model",
        },
        index_path=index_path,
        embedding_client=embedding_client,
    )

    assert len(retriever.vector_store.indexed_chunks) == len(DEFAULT_KNOWLEDGE_BASE)
    assert embedding_client.calls == [
        embedding_input(indexed_chunk.chunk)
        for indexed_chunk in retriever.vector_store.indexed_chunks
    ]


def test_runtime_creates_a_separate_retriever_for_external_references(
    index_path: Path,
) -> None:
    embedding_client = FakeEmbeddingClient()

    retriever = create_external_reference_gemini_vector_retriever(
        {
            "GEMINI_API_KEY": "test-key",
            "GEMINI_EMBEDDING_MODEL": "test-embedding-model",
        },
        index_path=index_path,
        embedding_client=embedding_client,
    )

    expected_chunks = load_external_reference_knowledge_base()
    assert len(retriever.vector_store.indexed_chunks) == len(expected_chunks)
    assert {item.chunk.document_id for item in retriever.vector_store.indexed_chunks} == {
        "consumer-remote-sales-v1",
        "consumer-exchange-return-v1",
    }
