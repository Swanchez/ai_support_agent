"""Retrieval contracts and a deterministic local implementation for learning RAG."""

import re
from dataclasses import dataclass
from typing import Protocol

from ai_support_agent.rag.knowledge_base import KnowledgeChunk
from ai_support_agent.rag.models import SourceType


WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


@dataclass(frozen=True)
class RetrievedChunk:
    """A knowledge fragment selected for a user question."""

    text: str
    score: float
    document_id: str
    title: str
    chunk_id: str = ""
    page_number: int | None = None
    source_type: SourceType = SourceType.INTERNAL_POLICY


class Retriever(Protocol):
    """Provider-independent interface for finding relevant knowledge chunks."""

    def retrieve(
        self,
        question: str,
        *,
        top_k: int,
        threshold: float,
        max_chunks_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks that satisfy the threshold."""


def tokenize(text: str) -> frozenset[str]:
    """Return normalized words for a deliberately simple keyword search."""

    return frozenset(word.lower() for word in WORD_PATTERN.findall(text))


@dataclass(frozen=True)
class KeywordRetriever:
    """Offline baseline retriever; it will later be replaced by embeddings."""

    chunks: tuple[KnowledgeChunk, ...]

    def retrieve(
        self,
        question: str,
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

        question_tokens = tokenize(question)
        if not question_tokens:
            return []

        results: list[RetrievedChunk] = []
        for chunk in self.chunks:
            chunk_tokens = tokenize(chunk.text)
            score = len(question_tokens & chunk_tokens) / len(question_tokens)
            if score >= threshold:
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


def limit_chunks_per_document(
    chunks: list[RetrievedChunk],
    *,
    top_k: int,
    max_chunks_per_document: int | None,
) -> list[RetrievedChunk]:
    """Prefer diverse documents while preserving descending similarity order."""

    ordered_chunks = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
    if max_chunks_per_document is None:
        return ordered_chunks[:top_k]

    selected_chunks: list[RetrievedChunk] = []
    selected_per_document: dict[str, int] = {}
    for chunk in ordered_chunks:
        selected_count = selected_per_document.get(chunk.document_id, 0)
        if selected_count >= max_chunks_per_document:
            continue
        selected_chunks.append(chunk)
        selected_per_document[chunk.document_id] = selected_count + 1
        if len(selected_chunks) == top_k:
            break
    return selected_chunks
