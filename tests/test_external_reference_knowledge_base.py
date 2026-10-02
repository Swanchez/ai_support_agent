from ai_support_agent.rag.knowledge_base import load_external_reference_knowledge_base
from ai_support_agent.rag.models import SourceType


def test_external_reference_collection_keeps_pdf_page_provenance() -> None:
    chunks = load_external_reference_knowledge_base()

    assert chunks
    assert {chunk.document_id for chunk in chunks} == {
        "consumer-remote-sales-v1",
        "consumer-exchange-return-v1",
        "consumer-remote-sales-rights-v1",
    }
    assert all(chunk.source_type is SourceType.EXTERNAL_REFERENCE for chunk in chunks)
    assert all(chunk.page_number is not None for chunk in chunks)
    assert all("#p" in chunk.chunk_id for chunk in chunks)
