"""Extracción y división de PDFs manteniendo página y sección como metadata."""

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
