"""Vista previa del splitter compartido por la ingesta de documentos."""

from pathlib import Path

from app.core.config import PROJECT_ROOT
from app.services.document_splitter import find_documents, split_document


DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"


def main() -> None:
    paths = find_documents(DOCUMENTS_DIR)
    if not paths:
        raise SystemExit(f"No se encontraron documentos en {DOCUMENTS_DIR}")

    total_chunks = 0
    for path in paths:
        chunks = split_document(path, DOCUMENTS_DIR)
        total_chunks += len(chunks)
        pages = sorted({chunk["metadata"]["page"] for chunk in chunks if "page" in chunk["metadata"]})
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
        print(f"  Primer fragmento ({chunks[0]['metadata'].get('section', 'sin sección')}):")
        print(f"  {chunks[0]['content'][:500].replace(chr(10), ' ')}")

    print(f"\nTotal: {len(paths)} documentos, {total_chunks} fragmentos.")


if __name__ == "__main__":
    main()
