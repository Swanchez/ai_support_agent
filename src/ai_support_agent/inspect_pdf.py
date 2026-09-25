"""Console command for checking whether a PDF has usable text before RAG indexing."""

import argparse
from pathlib import Path

from ai_support_agent.rag.pdf_ingestion import load_pdf_pages


def main() -> None:
    """Print a short text preview for every non-empty PDF page."""

    parser = argparse.ArgumentParser(
        description="Показать извлечённый текст PDF до добавления его в RAG."
    )
    parser.add_argument("path", type=Path, help="Путь к PDF-файлу")
    args = parser.parse_args()

    pages = load_pdf_pages(
        args.path,
        document_id=args.path.stem,
        title=args.path.stem.replace("_", " ").title(),
    )
    print(f"Извлечено страниц с текстом: {len(pages)}")
    for page in pages:
        preview = " ".join(page.text.split())[:300]
        print(f"\n[Страница {page.page_number}; символов: {len(page.text)}]")
        print(preview)


if __name__ == "__main__":
    main()
