"""Persistence for the small local RAG index used in this learning project."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ai_support_agent.rag.embeddings import EmbeddingClient
from ai_support_agent.rag.indexing import index_knowledge_base
from ai_support_agent.rag.knowledge_base import KnowledgeChunk
from ai_support_agent.rag.models import SourceType
from ai_support_agent.rag.vector_store import IndexedChunk, InMemoryVectorStore


@dataclass(frozen=True)
class IndexMetadata:
    """Properties that must match before a saved index can be reused."""

    embedding_model: str
    dimensions: int
    knowledge_base_fingerprint: str
    embedding_format_version: int = 1


def knowledge_base_fingerprint(chunks: Iterable[KnowledgeChunk]) -> str:
    """Return a stable fingerprint of the exact searchable source content."""

    source_data = [asdict(chunk) for chunk in chunks]
    serialized = json.dumps(
        source_data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def save_index(
    path: Path,
    index: InMemoryVectorStore,
    metadata: IndexMetadata,
) -> None:
    """Write a derived index atomically so an interrupted write keeps the old one."""

    payload = {
        "metadata": asdict(metadata),
        "indexed_chunks": [
            {
                "document_id": item.chunk.document_id,
                "chunk_id": item.chunk.chunk_id,
                "page_number": item.chunk.page_number,
                "source_type": item.chunk.source_type,
                "title": item.chunk.title,
                "text": item.chunk.text,
                "embedding": item.embedding,
            }
            for item in index.indexed_chunks
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_index(
    path: Path,
    expected_metadata: IndexMetadata,
) -> InMemoryVectorStore | None:
    """Load a compatible index; return None for absent, stale, or invalid cache data."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["metadata"] != asdict(expected_metadata):
            return None

        indexed_chunks = tuple(
            _indexed_chunk_from_payload(item, expected_metadata.dimensions)
            for item in payload["indexed_chunks"]
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    return InMemoryVectorStore(indexed_chunks)


def load_or_create_index(
    path: Path,
    chunks: Iterable[KnowledgeChunk],
    embedding_client: EmbeddingClient,
    metadata: IndexMetadata,
) -> InMemoryVectorStore:
    """Reuse a compatible cache or index the current knowledge base and save it."""

    cached_index = load_index(path, metadata)
    if cached_index is not None:
        return cached_index

    index = index_knowledge_base(chunks, embedding_client)
    save_index(path, index, metadata)
    return index


def _indexed_chunk_from_payload(item: Any, dimensions: int) -> IndexedChunk:
    """Validate one JSON record before it becomes part of the search index."""

    embedding = item["embedding"]
    if (
        not isinstance(item["document_id"], str)
        or not isinstance(item["chunk_id"], str)
        or (item["page_number"] is not None and (
            not isinstance(item["page_number"], int) or isinstance(item["page_number"], bool)
        ))
        or item["source_type"] not in {source_type.value for source_type in SourceType}
        or not isinstance(item["title"], str)
        or not isinstance(item["text"], str)
        or not isinstance(embedding, list)
        or len(embedding) != dimensions
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in embedding)
    ):
        raise ValueError("Invalid cached index record.")

    return IndexedChunk(
        chunk=KnowledgeChunk(
            document_id=item["document_id"],
            chunk_id=item["chunk_id"],
            page_number=item["page_number"],
            source_type=SourceType(item["source_type"]),
            title=item["title"],
            text=item["text"],
        ),
        embedding=tuple(float(value) for value in embedding),
    )
