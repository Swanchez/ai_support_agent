"""Shared RAG data contracts that do not load files or call external services."""

from dataclasses import dataclass
from enum import StrEnum


class SourceType(StrEnum):
    """How a source may be used when answering a support question."""

    INTERNAL_POLICY = "internal_policy"
    EXTERNAL_REFERENCE = "external_reference"


@dataclass(frozen=True)
class KnowledgeChunk:
    """One independently searchable fragment of a source document."""

    document_id: str
    title: str
    text: str
    chunk_id: str = ""
    page_number: int | None = None
    source_type: SourceType = SourceType.INTERNAL_POLICY
