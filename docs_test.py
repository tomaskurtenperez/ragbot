"""Vista previa del splitter compartido por la ingesta de PDFs."""

from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.services.document_splitter import split_pdf


DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main() -> None:
    pdf_paths = sorted(DOCUMENTS_DIR.rglob("*.pdf"))
    if not pdf_paths:
        raise SystemExit(f"No se encontraron PDFs en {DOCUMENTS_DIR}")

    total_chunks = 0
    for path in pdf_paths:
        chunks = split_pdf(path, DOCUMENTS_DIR)
        total_chunks += len(chunks)
        pages = sorted({chunk["metadata"]["page"] for chunk in chunks})
        sections = sorted(
            {
                chunk["metadata"]["section"]
                for chunk in chunks
                if chunk["metadata"].get("section")
            }
        )

        print(f"\n{path.name}")
        print(f"  Páginas con texto: {len(pages)}")
        print(f"  Fragmentos: {len(chunks)}")
        print(f"  Secciones detectadas: {len(sections)}")
        print(f"  Primer fragmento (página {chunks[0]['metadata']['page']}):")
        print(f"  {chunks[0]['content'][:500].replace(chr(10), ' ')}")

    print(f"\nTotal: {len(pdf_paths)} PDFs, {total_chunks} fragmentos.")


if __name__ == "__main__":
    main()
