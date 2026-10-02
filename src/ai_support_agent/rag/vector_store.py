"""A small in-memory vector index used to learn semantic retrieval."""

from dataclasses import dataclass
from math import sqrt
from typing import Protocol

from ai_support_agent.rag.embeddings import Embedding
from ai_support_agent.rag.knowledge_base import KnowledgeChunk
from ai_support_agent.rag.retriever import RetrievedChunk
from ai_support_agent.rag.retriever import limit_chunks_per_document


class VectorStore(Protocol):
    """Search stored embeddings without depending on their storage backend."""

    def search(
        self,
        query_embedding: Embedding,
        *,
        top_k: int = 3,
        threshold: float = 0.2,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]: ...


def cosine_similarity(left: Embedding, right: Embedding) -> float:
    """Measure how close two non-zero vectors point in the same direction."""

    if len(left) != len(right):
        raise ValueError("Embeddings must have the same dimensionality.")

    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


@dataclass(frozen=True)
class IndexedChunk:
    """A knowledge chunk together with its already calculated embedding."""

    chunk: KnowledgeChunk
    embedding: Embedding


@dataclass(frozen=True)
class InMemoryVectorStore:
    """Searches indexed chunks locally; it never calls an embedding API."""

    indexed_chunks: tuple[IndexedChunk, ...]

    def search(
        self,
        query_embedding: Embedding,
        *,
        top_k: int = 3,
        threshold: float = 0.2,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the closest stored chunks above the cosine-similarity threshold."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1.")
        if max_chunks_per_document is not None and max_chunks_per_document < 1:
            raise ValueError("max_chunks_per_document must be at least 1 when set.")

        results: list[RetrievedChunk] = []
        for indexed_chunk in self.indexed_chunks:
            score = cosine_similarity(query_embedding, indexed_chunk.embedding)
            if score >= threshold:
                chunk = indexed_chunk.chunk
                results.append(
                    RetrievedChunk(
                        text=chunk.text,
                        score=score,
                        document_id=chunk.document_id,
                        title=chunk.title,
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        source_type=chunk.source_type,
                    )
                )

        return limit_chunks_per_document(
            results,
            top_k=top_k,
            max_chunks_per_document=max_chunks_per_document,
        )
