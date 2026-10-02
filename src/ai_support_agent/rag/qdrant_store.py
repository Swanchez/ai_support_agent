"""Qdrant adapter and versioned collection initialization for semantic RAG."""

import hashlib
import json
from dataclasses import asdict, dataclass
from math import isfinite, nextafter
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from ai_support_agent.exceptions import VectorStoreError
from ai_support_agent.rag.embeddings import Embedding, EmbeddingClient
from ai_support_agent.rag.index_cache import IndexMetadata, load_index
from ai_support_agent.rag.indexing import index_knowledge_base
from ai_support_agent.rag.knowledge_base import KnowledgeChunk
from ai_support_agent.rag.models import SourceType
from ai_support_agent.rag.retriever import RetrievedChunk, limit_chunks_per_document


BACKEND_ERRORS = (UnexpectedResponse, ResponseHandlingException, httpx.HTTPError, OSError)


def collection_name(prefix: str, collection: str, metadata: IndexMetadata) -> str:
    """Keep incompatible models and source revisions in different collections."""
    serialized = json.dumps(asdict(metadata), sort_keys=True, separators=(",", ":"))
    revision = hashlib.sha256(serialized.encode()).hexdigest()[:24]
    return f"{prefix}_{collection}_{revision}"


def point_id(chunk: KnowledgeChunk) -> str:
    """Qdrant requires UUID or integer IDs; source IDs remain in the payload."""
    return str(uuid5(NAMESPACE_URL, f"{chunk.document_id}:{chunk.chunk_id}"))


def _validate_vector(vector: Embedding, dimensions: int) -> None:
    if len(vector) != dimensions or any(not isfinite(value) for value in vector):
        raise ValueError("Embedding must have the configured dimension and finite values.")
    if not any(vector):
        raise ValueError("Cosine similarity is undefined for a zero vector.")


@dataclass(frozen=True)
class QdrantVectorStore:
    """Rank stored chunks by cosine similarity and preserve the retrieval contract."""

    client: QdrantClient
    collection_name: str
    dimensions: int

    def close(self) -> None:
        self.client.close()

    def search(
        self,
        query_embedding: Embedding,
        *,
        top_k: int = 3,
        threshold: float = 0.2,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if max_chunks_per_document is not None and max_chunks_per_document < 1:
            raise ValueError("max_chunks_per_document must be at least 1 when set.")
        _validate_vector(query_embedding, self.dimensions)
        results: list[RetrievedChunk] = []
        offset = 0
        # Page until document diversity is satisfied; fixed overfetch can miss
        # a relevant second document after many chunks from the first one.
        page_size = max(top_k, 32) if max_chunks_per_document else top_k
        try:
            while True:
                points = self.client.query_points(
                    collection_name=self.collection_name,
                    query=list(query_embedding),
                    limit=page_size,
                    offset=offset,
                    # Qdrant excludes equality; our existing contract uses >=.
                    score_threshold=nextafter(threshold, float("-inf")),
                    with_payload=True,
                    with_vectors=False,
                ).points
                for point in points:
                    payload = point.payload or {}
                    results.append(RetrievedChunk(
                        text=payload["text"],
                        score=point.score,
                        document_id=payload["document_id"],
                        title=payload["title"],
                        chunk_id=payload["chunk_id"],
                        page_number=payload["page_number"],
                        source_type=SourceType(payload["source_type"]),
                    ))
                selected = limit_chunks_per_document(
                    results, top_k=top_k, max_chunks_per_document=max_chunks_per_document,
                )
                if len(selected) >= top_k or len(points) < page_size:
                    return selected
                offset += len(points)
        except (*BACKEND_ERRORS, KeyError, TypeError, ValueError) as error:
            raise VectorStoreError("Qdrant could not search the knowledge index.") from error


def load_or_create_qdrant_index(
    client: QdrantClient,
    name: str,
    chunks: tuple[KnowledgeChunk, ...],
    embedding_client: EmbeddingClient,
    metadata: IndexMetadata,
    *,
    import_path: Path,
) -> QdrantVectorStore:
    """Reuse a complete revision or import a compatible JSON cache before embedding."""
    expected_ids = {point_id(chunk) for chunk in chunks}
    if len(expected_ids) != len(chunks):
        raise ValueError("Knowledge chunks must have unique document_id/chunk_id pairs.")
    try:
        exists = client.collection_exists(name)
        if exists:
            params = client.get_collection(name).config.params.vectors
            if not isinstance(params, models.VectorParams) or params.size != metadata.dimensions or params.distance != models.Distance.COSINE:
                raise VectorStoreError("Qdrant collection has an incompatible vector configuration.")
            if client.count(name, exact=True).count == len(chunks):
                return QdrantVectorStore(client, name, metadata.dimensions)
        # JSON is read only for initial migration, never used for Qdrant search.
        index = load_index(import_path, metadata)
        if index is not None and {point_id(item.chunk) for item in index.indexed_chunks} != expected_ids:
            index = None
        if index is None:
            index = index_knowledge_base(chunks, embedding_client)
        for item in index.indexed_chunks:
            _validate_vector(item.embedding, metadata.dimensions)
        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(size=metadata.dimensions, distance=models.Distance.COSINE),
            )
        for start in range(0, len(index.indexed_chunks), 64):
            client.upsert(
                collection_name=name,
                points=[models.PointStruct(
                    id=point_id(item.chunk), vector=list(item.embedding), payload=asdict(item.chunk),
                ) for item in index.indexed_chunks[start:start + 64]],
                wait=True,
            )
        if client.count(name, exact=True).count != len(chunks):
            raise VectorStoreError("Qdrant index is incomplete; restart indexing before searching.")
        return QdrantVectorStore(client, name, metadata.dimensions)
    except BACKEND_ERRORS as error:
        raise VectorStoreError("Qdrant is unavailable or could not initialize the knowledge index.") from error
