"""Indexing functions that prepare knowledge chunks for semantic search."""

from collections.abc import Iterable

from ai_support_agent.rag.embeddings import BatchEmbeddingClient, Embedding, EmbeddingClient
from ai_support_agent.rag.knowledge_base import KnowledgeChunk
from ai_support_agent.rag.vector_store import IndexedChunk, InMemoryVectorStore


EMBEDDING_FORMAT_VERSION = 2
DEFAULT_EMBEDDING_BATCH_SIZE = 32


def embedding_input(chunk: KnowledgeChunk) -> str:
    """Add human-readable document context to the text represented by a vector."""

    return f"Документ: {chunk.title}\nТекст: {chunk.text}"


def index_knowledge_base(
    chunks: Iterable[KnowledgeChunk],
    embedding_client: EmbeddingClient,
    *,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> InMemoryVectorStore:
    """Embed chunks in bounded batches and return an in-memory vector index."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    chunks_tuple = tuple(chunks)
    indexed_chunks: list[IndexedChunk] = []
    for chunk_batch in _batches(chunks_tuple, batch_size):
        texts = [embedding_input(chunk) for chunk in chunk_batch]
        embeddings = _embed_batch(embedding_client, texts)
        if len(embeddings) != len(chunk_batch):
            raise ValueError("Embedding client returned a different number of vectors than texts.")
        indexed_chunks.extend(
            IndexedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunk_batch, embeddings)
        )
    return InMemoryVectorStore(tuple(indexed_chunks))


def _embed_batch(embedding_client: EmbeddingClient, texts: list[str]) -> list[Embedding]:
    """Use a batch-capable client when available, retaining simple test-client support."""

    if isinstance(embedding_client, BatchEmbeddingClient):
        return embedding_client.embed_many(texts)
    return [embedding_client.embed(text) for text in texts]


def _batches(
    values: tuple[KnowledgeChunk, ...], batch_size: int
) -> Iterable[tuple[KnowledgeChunk, ...]]:
    """Yield fixed-size tuples without copying the entire source collection again."""

    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]
