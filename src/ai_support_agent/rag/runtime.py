"""Application assembly for the local Gemini-backed semantic retriever."""

from pathlib import Path
from typing import Mapping

from ai_support_agent.config import (
    PROJECT_ROOT,
    load_environment,
    load_gemini_embedding_config,
)
from ai_support_agent.rag.embeddings import (
    GEMINI_EMBEDDING_DIMENSIONS,
    EmbeddingClient,
    GeminiEmbeddingClient,
)
from ai_support_agent.rag.index_cache import (
    IndexMetadata,
    knowledge_base_fingerprint,
    load_or_create_index,
)
from ai_support_agent.rag.indexing import EMBEDDING_FORMAT_VERSION
from ai_support_agent.rag.knowledge_base import (
    DEFAULT_KNOWLEDGE_BASE,
    KnowledgeChunk,
    load_external_reference_knowledge_base,
)
from ai_support_agent.rag.vector_retriever import VectorRetriever


RAG_INDEX_PATH = PROJECT_ROOT / "data" / "rag_index.json"
EXTERNAL_REFERENCE_RAG_INDEX_PATH = PROJECT_ROOT / "data" / "external_reference_rag_index.json"


def create_gemini_vector_retriever(
    env: Mapping[str, str] | None = None,
    *,
    index_path: Path = RAG_INDEX_PATH,
    embedding_client: EmbeddingClient | None = None,
) -> VectorRetriever:
    """Create a cached semantic retriever, using Gemini only when an index is stale."""

    return _create_gemini_vector_retriever_for_chunks(
        DEFAULT_KNOWLEDGE_BASE,
        env,
        index_path=index_path,
        embedding_client=embedding_client,
    )


def create_external_reference_gemini_vector_retriever(
    env: Mapping[str, str] | None = None,
    *,
    index_path: Path = EXTERNAL_REFERENCE_RAG_INDEX_PATH,
    embedding_client: EmbeddingClient | None = None,
) -> VectorRetriever:
    """Create a separately cached retriever for non-authoritative external PDFs."""

    return _create_gemini_vector_retriever_for_chunks(
        load_external_reference_knowledge_base(),
        env,
        index_path=index_path,
        embedding_client=embedding_client,
    )


def _create_gemini_vector_retriever_for_chunks(
    chunks: tuple[KnowledgeChunk, ...],
    env: Mapping[str, str] | None,
    *,
    index_path: Path,
    embedding_client: EmbeddingClient | None,
) -> VectorRetriever:
    """Build one vector retriever from a supplied collection and its own cache file."""

    environment = load_environment(env)
    embedding_config = load_gemini_embedding_config(environment)
    client = embedding_client or GeminiEmbeddingClient(embedding_config)
    metadata = IndexMetadata(
        embedding_model=embedding_config.model,
        dimensions=GEMINI_EMBEDDING_DIMENSIONS,
        knowledge_base_fingerprint=knowledge_base_fingerprint(chunks),
        embedding_format_version=EMBEDDING_FORMAT_VERSION,
    )
    vector_store = load_or_create_index(
        index_path,
        chunks,
        client,
        metadata,
    )
    return VectorRetriever(client, vector_store)
