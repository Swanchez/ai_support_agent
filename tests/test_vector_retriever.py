from dataclasses import dataclass, field

from ai_support_agent.rag.embeddings import Embedding
from ai_support_agent.rag.knowledge_base import KnowledgeChunk
from ai_support_agent.rag.vector_retriever import VectorRetriever
from ai_support_agent.rag.vector_store import IndexedChunk, InMemoryVectorStore


@dataclass
class FakeEmbeddingClient:
    embedding: Embedding
    received_texts: list[str] = field(default_factory=list)

    def embed(self, text: str) -> Embedding:
        self.received_texts.append(text)
        return self.embedding


def test_vector_retriever_embeds_only_the_question_then_searches_index() -> None:
    chunk = KnowledgeChunk("refund-v1", "Возврат", "Возврат денег")
    store = InMemoryVectorStore((IndexedChunk(chunk, (1.0, 0.0)),))
    embedding_client = FakeEmbeddingClient((1.0, 0.0))
    retriever = VectorRetriever(embedding_client, store)

    results = retriever.retrieve("Когда зачислят средства?", top_k=3, threshold=0.2)

    assert embedding_client.received_texts == ["Когда зачислят средства?"]
    assert [result.document_id for result in results] == ["refund-v1"]
