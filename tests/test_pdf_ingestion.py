from dataclasses import dataclass
from pathlib import Path

import pytest

from ai_support_agent.rag.pdf_ingestion import PdfTextExtractionError, load_pdf_pages
from ai_support_agent.rag.models import SourceType


@dataclass
class FakePage:
    text: str | None

    def extract_text(self) -> str | None:
        return self.text


@dataclass
class FakeReader:
    pages: list[FakePage]


def test_load_pdf_pages_keeps_page_numbers_and_skips_empty_pages() -> None:
    reader = FakeReader([FakePage("First page"), FakePage("  "), FakePage("Third page")])

    pages = load_pdf_pages(
        Path("document.pdf"),
        document_id="consumer-rights-v1",
        title="Consumer rights",
        reader_factory=lambda _: reader,
    )

    assert [(page.page_number, page.text) for page in pages] == [
        (1, "First page"),
        (3, "Third page"),
    ]
    assert all(page.document_id == "consumer-rights-v1" for page in pages)
    assert all(page.source_type is SourceType.EXTERNAL_REFERENCE for page in pages)


def test_load_pdf_pages_explains_when_no_text_layer_exists() -> None:
    reader = FakeReader([FakePage(None), FakePage(" ")])

    with pytest.raises(PdfTextExtractionError, match="needs OCR"):
        load_pdf_pages(
            Path("scan.pdf"),
            document_id="scan-v1",
            title="Scan",
            reader_factory=lambda _: reader,
        )
