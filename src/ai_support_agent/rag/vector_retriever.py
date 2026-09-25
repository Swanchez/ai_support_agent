"""Semantic retriever built from an embedding client and a vector index."""

from dataclasses import dataclass

from ai_support_agent.rag.embeddings import EmbeddingClient
from ai_support_agent.rag.retriever import RetrievedChunk
from ai_support_agent.rag.vector_store import InMemoryVectorStore


@dataclass(frozen=True)
class VectorRetriever:
    """Embeds a question, then delegates local ranking to the vector store."""

    embedding_client: EmbeddingClient
    vector_store: InMemoryVectorStore

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 3,
        threshold: float = 0.2,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        """Find chunks that are semantically close to the question."""

        question_embedding = self.embedding_client.embed(question)
        return self.vector_store.search(
            question_embedding,
            top_k=top_k,
            threshold=threshold,
            max_chunks_per_document=max_chunks_per_document,
        )
