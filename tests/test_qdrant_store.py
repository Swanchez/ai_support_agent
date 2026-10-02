from dataclasses import replace
from contextlib import closing
from pathlib import Path

import httpx
import pytest
from qdrant_client import QdrantClient, models

from ai_support_agent.config import load_vector_store_config
from ai_support_agent.exceptions import ConfigurationError, VectorStoreError
from ai_support_agent.rag.index_cache import IndexMetadata, knowledge_base_fingerprint, save_index
from ai_support_agent.rag.models import KnowledgeChunk, SourceType
from ai_support_agent.rag.qdrant_store import QdrantVectorStore, collection_name, load_or_create_qdrant_index, point_id
from ai_support_agent.rag.vector_store import IndexedChunk, InMemoryVectorStore


CHUNKS = (
    KnowledgeChunk("refund", "Refund", "Money timing", "refund-1"),
    KnowledgeChunk("refund", "Refund", "Warehouse", "refund-2"),
    KnowledgeChunk("delivery", "Delivery", "Delivery timing", "delivery-1", 2, SourceType.EXTERNAL_REFERENCE),
)
VECTORS = ((1.0, 0.0), (0.99, 0.1), (0.8, 0.6))


@pytest.fixture
def index_path():
    path = Path(__file__).with_name("_qdrant_test_index.json")
    path.unlink(missing_ok=True)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


class Embeddings:
    def __init__(self):
        self.calls = 0

    def embed(self, text):
        result = VECTORS[self.calls]
        self.calls += 1
        return result


def metadata(chunks=CHUNKS):
    return IndexMetadata("test-model", 2, knowledge_base_fingerprint(chunks), 2)


def test_qdrant_search_matches_cosine_and_document_limit(index_path: Path):
    with closing(QdrantClient(":memory:")) as client:
        store = load_or_create_qdrant_index(client, "test", CHUNKS, Embeddings(), metadata(), import_path=index_path)
        results = store.search((1.0, 0.0), threshold=0.7, top_k=2, max_chunks_per_document=1)
        assert [item.chunk_id for item in results] == ["refund-1", "delivery-1"]
        assert [item.score for item in results] == pytest.approx([1.0, 0.8])
        assert results[1].page_number == 2
        assert results[1].source_type is SourceType.EXTERNAL_REFERENCE
        assert store.search((1.0, 0.0), threshold=1.0)[0].text == "Money timing"
        assert store.search((-1.0, 0.0), threshold=0.7) == []


def test_import_and_restart_do_not_call_embedding_provider(index_path: Path):
    path = index_path
    save_index(path, InMemoryVectorStore(tuple(IndexedChunk(c, v) for c, v in zip(CHUNKS, VECTORS))), metadata())
    embedding = Embeddings()
    with closing(QdrantClient(":memory:")) as client:
        load_or_create_qdrant_index(client, "test", CHUNKS, embedding, metadata(), import_path=path)
        path.unlink()
        load_or_create_qdrant_index(client, "test", CHUNKS, embedding, metadata(), import_path=path)
        assert embedding.calls == 0
        assert client.count("test", exact=True).count == 3


def test_stale_cache_is_not_imported(index_path: Path):
    path = index_path
    save_index(path, InMemoryVectorStore(tuple(IndexedChunk(c, v) for c, v in zip(CHUNKS, VECTORS))), replace(metadata(), embedding_model="old-model"))
    embedding = Embeddings()
    with closing(QdrantClient(":memory:")) as client:
        load_or_create_qdrant_index(client, "new", CHUNKS, embedding, metadata(), import_path=path)
    assert embedding.calls == 3


def test_partial_upload_can_be_completed_after_restart(index_path: Path):
    with closing(QdrantClient(":memory:")) as client:
        client.create_collection("partial", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
        client.upsert("partial", [models.PointStruct(id=point_id(CHUNKS[0]), vector=list(VECTORS[0]), payload={})])
        store = load_or_create_qdrant_index(client, "partial", CHUNKS, Embeddings(), metadata(), import_path=index_path)
        assert len(store.search((1.0, 0.0), threshold=0.0)) == 3


def test_new_model_or_documents_use_a_new_collection():
    name = collection_name("support", "internal", metadata())
    assert name != collection_name("support", "external", metadata())
    assert name != collection_name("support", "internal", replace(metadata(), embedding_model="another"))
    assert name != collection_name("support", "internal", replace(metadata(), knowledge_base_fingerprint="changed"))


def test_document_limit_fetches_beyond_first_page(index_path: Path):
    chunks = tuple(KnowledgeChunk("a", "A", "Text", f"a-{i}") for i in range(40)) + (CHUNKS[2],)
    path = index_path
    index = InMemoryVectorStore(tuple(IndexedChunk(c, (1.0, 0.0)) for c in chunks[:-1]) + (IndexedChunk(chunks[-1], VECTORS[2]),))
    save_index(path, index, metadata(chunks))
    with closing(QdrantClient(":memory:")) as client:
        store = load_or_create_qdrant_index(client, "test", chunks, Embeddings(), metadata(chunks), import_path=path)
        assert [r.document_id for r in store.search((1.0, 0.0), top_k=2, threshold=0.7, max_chunks_per_document=1)] == ["a", "delivery"]


def test_qdrant_failure_is_not_reported_as_no_results():
    class FailingClient:
        def query_points(self, **kwargs):
            raise httpx.ConnectError("private connection details")

    store = QdrantVectorStore(FailingClient(), "test", 2)
    with pytest.raises(VectorStoreError, match="could not search"):
        store.search((1.0, 0.0))


@pytest.mark.parametrize("settings", [
    {"RAG_VECTOR_BACKEND": "unknown"},
    {"RAG_VECTOR_BACKEND": "qdrant", "QDRANT_URL": "not-url"},
    {"RAG_VECTOR_BACKEND": "qdrant", "QDRANT_COLLECTION_PREFIX": "bad/name"},
    {"QDRANT_TIMEOUT_SECONDS": "0"},
])
def test_invalid_backend_configuration_is_rejected(settings):
    with pytest.raises(ConfigurationError):
        load_vector_store_config(settings)
