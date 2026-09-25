"""Extraction and deterministic chunking of local Markdown knowledge documents."""

from dataclasses import dataclass
from pathlib import Path

from ai_support_agent.rag.models import KnowledgeChunk, SourceType


DEFAULT_CHUNK_SIZE = 320
DEFAULT_CHUNK_OVERLAP = 40


@dataclass(frozen=True)
class SourceDocument:
    """Plain text and metadata extracted from one source document."""

    document_id: str
    title: str
    text: str
    page_number: int | None = None
    source_type: SourceType = SourceType.INTERNAL_POLICY


def load_markdown_document(path: Path, document_id: str) -> SourceDocument:
    """Read a UTF-8 Markdown file and use its first level-one heading as the title."""

    lines = path.read_text(encoding="utf-8").splitlines()
    fallback_title = path.stem.replace("_", " ").title()
    title = fallback_title
    body_lines: list[str] = []
    for line in lines:
        if line.startswith("# ") and title == fallback_title:
            title = line.removeprefix("# ").strip()
            continue
        body_lines.append(line)

    return SourceDocument(
        document_id=document_id,
        title=title,
        text="\n".join(body_lines),
    )


def chunk_document(
    document: SourceDocument,
    *,
    max_characters: int = DEFAULT_CHUNK_SIZE,
    overlap_characters: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[KnowledgeChunk, ...]:
    """Split document text at word boundaries and assign stable sequential chunk IDs."""

    chunks = [
        chunk
        for section in markdown_sections(document.text)
        for chunk in split_text(
            section,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )
    ]
    return tuple(
        KnowledgeChunk(
            document_id=document.document_id,
            title=document.title,
            text=text,
            chunk_id=_chunk_id(document, position),
            page_number=document.page_number,
            source_type=document.source_type,
        )
        for position, text in enumerate(chunks, start=1)
    )


def _chunk_id(document: SourceDocument, position: int) -> str:
    """Keep existing Markdown IDs while making PDF page chunks globally unique."""

    if document.page_number is None:
        return f"{document.document_id}#{position}"
    return f"{document.document_id}#p{document.page_number}-c{position}"


def split_text(
    text: str,
    *,
    max_characters: int = DEFAULT_CHUNK_SIZE,
    overlap_characters: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Normalize whitespace, then make overlapping chunks without splitting words."""

    if max_characters < 1:
        raise ValueError("max_characters must be at least 1.")
    if not 0 <= overlap_characters < max_characters:
        raise ValueError("overlap_characters must be at least 0 and smaller than max_characters.")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []
    for word in words:
        candidate = " ".join([*current_words, word])
        if current_words and len(candidate) > max_characters:
            chunks.append(" ".join(current_words))
            current_words = _overlap_words(current_words, overlap_characters)
            while current_words and len(" ".join([*current_words, word])) > max_characters:
                current_words.pop(0)
        current_words.append(word)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def markdown_sections(markdown: str) -> list[str]:
    """Return normalized Markdown sections, keeping a level-two heading with its text."""

    sections: list[str] = []
    current_lines: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## ") and current_lines:
            section = " ".join(" ".join(current_lines).split())
            if section:
                sections.append(section)
            current_lines = [line]
        else:
            current_lines.append(line)

    final_section = " ".join(" ".join(current_lines).split())
    if final_section:
        sections.append(final_section)
    return sections


def _overlap_words(words: list[str], overlap_characters: int) -> list[str]:
    """Keep as many trailing whole words as fit into the requested overlap."""

    if overlap_characters == 0:
        return []

    overlap: list[str] = []
    for word in reversed(words):
        candidate = " ".join([word, *overlap])
        if overlap and len(candidate) > overlap_characters:
            break
        overlap.insert(0, word)
    return overlap
