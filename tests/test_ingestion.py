from pathlib import Path

import pytest

from ai_support_agent.rag.ingestion import (
    SourceDocument,
    chunk_document,
    load_markdown_document,
    markdown_sections,
    split_text,
)


def test_load_markdown_document_extracts_heading_and_body() -> None:
    document = load_markdown_document(
        Path("knowledge/refund_policy.md"),
        "refund-policy-v1",
    )

    assert document.document_id == "refund-policy-v1"
    assert document.title == "Правила возврата"
    assert "3–7 рабочих дней" in document.text
    assert "# Правила возврата" not in document.text


def test_split_text_preserves_words_and_applies_overlap() -> None:
    chunks = split_text(
        "один два три четыре пять шесть семь",
        max_characters=17,
        overlap_characters=5,
    )

    assert chunks == ["один два три", "три четыре пять", "пять шесть семь"]


def test_chunk_document_assigns_stable_chunk_ids() -> None:
    document = SourceDocument("refund-policy-v1", "Возврат", "один два три четыре")

    chunks = chunk_document(document, max_characters=10, overlap_characters=0)

    assert [chunk.chunk_id for chunk in chunks] == [
        "refund-policy-v1#1",
        "refund-policy-v1#2",
    ]
    assert [chunk.document_id for chunk in chunks] == [
        "refund-policy-v1",
        "refund-policy-v1",
    ]


def test_chunk_document_includes_pdf_page_number_in_chunk_id() -> None:
    document = SourceDocument(
        "consumer-rights-v1",
        "Права потребителя",
        "один два три четыре",
        page_number=2,
    )

    chunks = chunk_document(document, max_characters=10, overlap_characters=0)

    assert [chunk.chunk_id for chunk in chunks] == [
        "consumer-rights-v1#p2-c1",
        "consumer-rights-v1#p2-c2",
    ]
    assert all(chunk.page_number == 2 for chunk in chunks)


def test_markdown_sections_keep_each_heading_with_its_facts() -> None:
    sections = markdown_sections("## First\nFirst fact.\n\n## Second\nSecond fact.")

    assert sections == ["## First First fact.", "## Second Second fact."]


@pytest.mark.parametrize(
    ("max_characters", "overlap_characters"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_split_text_rejects_invalid_chunk_settings(
    max_characters: int,
    overlap_characters: int,
) -> None:
    with pytest.raises(ValueError):
        split_text("текст", max_characters=max_characters, overlap_characters=overlap_characters)
