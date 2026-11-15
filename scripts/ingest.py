"""Indexa los documentos de data/documents (PDF y .txt) con el modelo local de Hugging Face."""

import hashlib

import chromadb
from dotenv import load_dotenv

from app.core.config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_DB_PATH,
    HF_EMBEDDING_MODEL,
    PROJECT_ROOT,
)
from app.services.document_splitter import find_documents, split_document
from app.services.embedding_service import generate_embeddings, get_embedding_model


DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"
EMBEDDING_BATCH_SIZE = 32
CHROMA_BATCH_SIZE = 256


def make_id(chunk: dict) -> str:
    metadata = chunk["metadata"]
    identity = (
        f"{metadata['document_path']}|{metadata.get('page', '')}|"
        f"{metadata['chunk_index']}|{chunk['content']}"
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def ingest() -> None:
    load_dotenv()
    if not DOCUMENTS_DIR.is_dir():
        raise SystemExit(f"No existe la carpeta de documentos: {DOCUMENTS_DIR}")

    document_paths = find_documents(DOCUMENTS_DIR)
    if not document_paths:
        raise SystemExit(f"No se encontraron documentos en {DOCUMENTS_DIR}")

    chunks: list[dict] = []
    for path in document_paths:
        document_chunks = split_document(path, DOCUMENTS_DIR)
        chunks.extend(document_chunks)
        print(f"{path.name}: {len(document_chunks)} fragmentos")

    if not chunks:
        raise SystemExit("No se extrajo texto indexable de los documentos.")

    model = get_embedding_model()
    print(
        f"Modelo local: {HF_EMBEDDING_MODEL} "
        f"({model.get_sentence_embedding_dimension()} dimensiones)"
    )
    vectors = generate_embeddings(
        [f"passage: {chunk['content']}" for chunk in chunks],
        batch_size=EMBEDDING_BATCH_SIZE,
    )
    if len(vectors) != len(chunks):
        raise RuntimeError("La cantidad de embeddings no coincide con los fragmentos.")

    chroma = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = chroma.get_or_create_collection(name=CHROMA_COLLECTION_NAME)

    # Conserva la base anterior si extracción o generación de vectores falla.
    existing_ids = collection.get(include=["metadatas"])["ids"]
    for start in range(0, len(existing_ids), CHROMA_BATCH_SIZE):
        collection.delete(ids=existing_ids[start : start + CHROMA_BATCH_SIZE])

    ids = [make_id(chunk) for chunk in chunks]
    contents = [chunk["content"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    for start in range(0, len(chunks), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=contents[start:end],
            embeddings=vectors[start:end].tolist(),
            metadatas=metadatas[start:end],
        )
        print(f"Indexados: {min(end, len(chunks))}/{len(chunks)}")

    print(
        f"Ingesta lista: {collection.count()} fragmentos en "
        f"'{CHROMA_COLLECTION_NAME}' ({CHROMA_DB_PATH})."
    )


if __name__ == "__main__":
    ingest()
