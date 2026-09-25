from dataclasses import dataclass, field

from ai_support_agent.rag.embeddings import Embedding
from ai_support_agent.rag.indexing import embedding_input, index_knowledge_base
from ai_support_agent.rag.knowledge_base import KnowledgeChunk


@dataclass
class FakeEmbeddingClient:
    received_texts: list[str] = field(default_factory=list)
    received_batches: list[list[str]] = field(default_factory=list)

    def embed(self, text: str) -> Embedding:
        self.received_texts.append(text)
        return (float(len(text)), 1.0)

    def embed_many(self, texts: list[str]) -> list[Embedding]:
        self.received_batches.append(texts)
        return [self.embed(text) for text in texts]


def test_index_knowledge_base_embeds_every_chunk_once_with_its_text() -> None:
    chunks = (
        KnowledgeChunk("refund-v1", "Возврат", "Возврат денег"),
        KnowledgeChunk("delivery-v1", "Доставка", "Срок доставки"),
    )
    embedding_client = FakeEmbeddingClient()

    index = index_knowledge_base(chunks, embedding_client)

    assert embedding_client.received_texts == [embedding_input(chunk) for chunk in chunks]
    assert embedding_client.received_batches == [[embedding_input(chunk) for chunk in chunks]]
    assert [item.chunk.document_id for item in index.indexed_chunks] == [
        "refund-v1",
        "delivery-v1",
    ]
    assert index.indexed_chunks[0].embedding == (
        float(len(embedding_input(chunks[0]))),
        1.0,
    )


def test_index_knowledge_base_splits_large_collections_into_bounded_batches() -> None:
    chunks = tuple(
        KnowledgeChunk(f"document-{number}", "Title", f"Text {number}")
        for number in range(3)
    )
    embedding_client = FakeEmbeddingClient()

    index_knowledge_base(chunks, embedding_client, batch_size=2)

    assert embedding_client.received_batches == [
        [embedding_input(chunks[0]), embedding_input(chunks[1])],
        [embedding_input(chunks[2])],
    ]
