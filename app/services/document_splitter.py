"""Extracción y división de documentos (PDF y texto plano) con metadata de página/sección."""

import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _normalize_page(text: str) -> str:
    text = text.replace("\x00", "").replace("\u00ad", "")
    # Une palabras que quedaron separadas por un guion al final de línea.
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 140 or len(line.split()) > 18:
        return False
    if line.endswith((".", ",", ";", ":", "?", "!")):
        return False

    numbered = bool(re.match(r"^\d+(?:\.\d+)*[.)]?\s+\S", line))
    letters = [character for character in line if character.isalpha()]
    all_caps = len(letters) >= 4 and all(character.isupper() for character in letters)
    title_case = len(line.split()) <= 12 and all(
        not word[0].isalpha() or word[0].isupper()
        for word in line.split()
        if word
    )
    return numbered or all_caps or title_case


def _section_at(text: str, offset: int) -> str:
    section = ""
    for line in text[:offset].splitlines():
        if _is_heading(line):
            section = line.strip()
    return section


def split_pdf(path: Path, documents_root: Path) -> list[dict]:
    """Devuelve chunks con IDs y metadata estables para indexar en Chroma."""
    reader = PdfReader(str(path))
    relative_path = path.relative_to(documents_root).as_posix()
    chunks: list[dict] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = _normalize_page(page.extract_text() or "")
        if not page_text:
            continue

        base_metadata = {
            "filename": path.name,
            "document_path": relative_path,
            "page": page_number,
        }
        split_documents = _splitter.create_documents(
            [page_text], metadatas=[base_metadata]
        )

        for chunk_index, document in enumerate(split_documents):
            start_index = int(document.metadata.pop("start_index", 0))
            section = _section_at(page_text, start_index)
            metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "start_index": start_index,
            }
            if section:
                metadata["section"] = section[:200]

            chunks.append({"content": document.page_content, "metadata": metadata})

    if not chunks:
        raise ValueError(
            f"No se pudo extraer texto de {path.name}; puede ser un PDF escaneado."
        )
    return chunks


# Encabezado de sección de un RFC: empieza en la columna 0, p. ej.
# "15.3.1.  200 OK" o "Appendix A.  Collected ABNF".
_RFC_HEADING = re.compile(
    r"^(?:\d+(?:\.\d+)*\.|Appendix [A-Z](?:\.\d+)*\.)\s+\S.*$",
    re.MULTILINE,
)
# Encabezados y pies de página de los RFC con formato antiguo (paginados).
_RFC_PAGE_FURNITURE = re.compile(
    r"^(?:RFC \d+\s{2,}.*\d{4}|.*\[Page \d+\])[ \t]*$\n?",
    re.MULTILINE,
)


def _normalize_text_document(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\f", "\n")
    text = _RFC_PAGE_FURNITURE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text_file(path: Path, documents_root: Path) -> list[dict]:
    """Divide un documento de texto (p. ej. un RFC) por secciones y luego en chunks."""
    text = _normalize_text_document(path.read_text(encoding="utf-8-sig"))
    relative_path = path.relative_to(documents_root).as_posix()

    headings = list(_RFC_HEADING.finditer(text))
    if headings:
        bounds = [(0, headings[0].start(), "Front matter")] + [
            (
                match.start(),
                headings[index + 1].start() if index + 1 < len(headings) else len(text),
                match.group(0).strip(),
            )
            for index, match in enumerate(headings)
        ]
    else:
        bounds = [(0, len(text), "")]

    chunks: list[dict] = []
    for start, end, section in bounds:
        section_text = text[start:end]
        if section == "Front matter":
            # Conserva el abstract pero descarta el índice: solo repite títulos
            # de sección y genera ruido en la búsqueda léxica.
            section_text = section_text.split("\nTable of Contents", 1)[0]
        section_text = section_text.strip()
        if not section_text:
            continue
        base_metadata = {"filename": path.name, "document_path": relative_path}
        if section:
            base_metadata["section"] = re.sub(r"\s+", " ", section)[:200]
        for document in _splitter.create_documents(
            [section_text], metadatas=[base_metadata]
        ):
            offset = int(document.metadata.pop("start_index", 0))
            chunks.append(
                {
                    "content": document.page_content,
                    "metadata": {
                        **document.metadata,
                        "chunk_index": len(chunks),
                        "start_index": start + offset,
                    },
                }
            )

    if not chunks:
        raise ValueError(f"{path.name} no contiene texto indexable.")
    return chunks


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def split_document(path: Path, documents_root: Path) -> list[dict]:
    """Elige el splitter según la extensión del archivo."""
    if path.suffix.lower() == ".pdf":
        return split_pdf(path, documents_root)
    if path.suffix.lower() == ".txt":
        return split_text_file(path, documents_root)
    raise ValueError(f"Formato no soportado: {path.name}")


def find_documents(documents_root: Path) -> list[Path]:
    return sorted(
        path
        for path in documents_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
