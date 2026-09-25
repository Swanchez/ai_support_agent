"""Extraction of text-layer PDF pages into the same documents used by Markdown ingestion."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from ai_support_agent.rag.ingestion import SourceDocument
from ai_support_agent.rag.models import SourceType


class PdfPage(Protocol):
    """The small part of a pypdf page that our loader needs."""

    def extract_text(self) -> str | None:
        """Return text from the page's text layer, if it exists."""


class PdfReader(Protocol):
    """The small part of a pypdf reader that our loader needs."""

    pages: list[PdfPage]


class PdfTextExtractionError(RuntimeError):
    """A PDF did not provide usable text-layer content for ingestion."""


def load_pdf_pages(
    path: Path,
    *,
    document_id: str,
    title: str,
    source_type: SourceType = SourceType.EXTERNAL_REFERENCE,
    reader_factory: Callable[[Path], PdfReader] | None = None,
) -> tuple[SourceDocument, ...]:
    """Extract non-empty text pages as independent source documents with page numbers."""

    if reader_factory is None:
        try:
            from pypdf import PdfReader as PypdfReader
        except ImportError as error:
            raise PdfTextExtractionError(
                "pypdf is not installed. Run: python -m pip install -e \".[dev]\""
            ) from error
        reader_factory = PypdfReader

    reader = reader_factory(path)
    pages = tuple(
        SourceDocument(
            document_id=document_id,
            title=title,
            text=text,
            page_number=page_number,
            source_type=source_type,
        )
        for page_number, page in enumerate(reader.pages, start=1)
        if (text := (page.extract_text() or "").strip())
    )
    if not pages:
        raise PdfTextExtractionError(
            "PDF has no extractable text. It may be a scanned document that needs OCR."
        )
    return pages
