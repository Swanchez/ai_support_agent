"""Loading the project's searchable knowledge chunks from source documents."""

from pathlib import Path

from ai_support_agent.config import PROJECT_ROOT
from ai_support_agent.rag.models import KnowledgeChunk


KNOWLEDGE_DIRECTORY = PROJECT_ROOT / "knowledge"
DEFAULT_SOURCE_DOCUMENTS: tuple[tuple[str, Path], ...] = (
    ("refund-policy-v1", KNOWLEDGE_DIRECTORY / "refund_policy.md"),
    ("delivery-policy-v1", KNOWLEDGE_DIRECTORY / "delivery_policy.md"),
    ("warranty-policy-v1", KNOWLEDGE_DIRECTORY / "warranty_policy.md"),
)
PDF_DIRECTORY = KNOWLEDGE_DIRECTORY / "pdf"
EXTERNAL_REFERENCE_DOCUMENTS: tuple[tuple[str, Path, str], ...] = (
    (
        "consumer-remote-sales-v1",
        PDF_DIRECTORY / "remote_sales_return.pdf",
        "Возврат товара при дистанционной продаже",
    ),
    (
        "consumer-exchange-return-v1",
        PDF_DIRECTORY / "product_exchange_return.pdf",
        "Обмен и возврат непродовольственного товара",
    ),
    (
        "consumer-remote-sales-rights-v1",
        PDF_DIRECTORY / "О правах потребителя при дистанционном способе продажи товаров.pdf",
        "Права потребителя при дистанционной продаже товаров",
    ),
)


def load_default_knowledge_base() -> tuple[KnowledgeChunk, ...]:
    """Read and chunk every Markdown document that belongs to the learning knowledge base."""

    from ai_support_agent.rag.ingestion import chunk_document, load_markdown_document

    chunks: list[KnowledgeChunk] = []
    for document_id, path in DEFAULT_SOURCE_DOCUMENTS:
        document = load_markdown_document(path, document_id)
        chunks.extend(chunk_document(document))
    return tuple(chunks)


def load_external_reference_knowledge_base() -> tuple[KnowledgeChunk, ...]:
    """Load PDF reference material into a separate, non-production collection."""

    from ai_support_agent.rag.ingestion import chunk_document
    from ai_support_agent.rag.pdf_ingestion import load_pdf_pages

    chunks: list[KnowledgeChunk] = []
    for document_id, path, title in EXTERNAL_REFERENCE_DOCUMENTS:
        for page in load_pdf_pages(path, document_id=document_id, title=title):
            chunks.extend(chunk_document(page))
    return tuple(chunks)


DEFAULT_KNOWLEDGE_BASE = load_default_knowledge_base()
