import re

import chromadb
from rank_bm25 import BM25Okapi

from app.core.config import CHROMA_COLLECTION_NAME, CHROMA_DB_PATH
from app.services.embedding_service import generate_query_embedding


_collection = None
_bm25_cache = None


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def get_collection():
    """Abre Chroma sólo cuando una consulta necesita recuperar datos."""
    global _collection
    if _collection is None:
        chroma = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        _collection = chroma.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    return _collection


def build_bm25_index(collection):
    """Construye o devuelve el índice léxico cacheado de la colección Chroma."""
    global _bm25_cache
    document_count = collection.count()

    if document_count == 0:
        _bm25_cache = (0, None, [], [], [])
        return _bm25_cache[1:]

    if _bm25_cache is not None and _bm25_cache[0] == document_count:
        return _bm25_cache[1:]

    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or [{} for _ in documents]
    ids = data.get("ids") or []
    tokenized_documents = [tokenize(document or "") for document in documents]

    if not tokenized_documents:
        _bm25_cache = (document_count, None, [], [], [])
    else:
        bm25 = BM25Okapi(tokenized_documents)
        _bm25_cache = (document_count, bm25, documents, metadatas, ids)

    return _bm25_cache[1:]


def warm_retriever() -> None:
    """Abre Chroma y prepara BM25 para que la primera consulta no lo haga."""
    collection = get_collection()
    build_bm25_index(collection)


async def search_similar(question: str, limit: int = 5) -> list[dict]:
    collection = get_collection()
    document_count = collection.count()
    if document_count == 0:
        return []

    query_embedding = await generate_query_embedding(question)
    candidate_limit = min(document_count, max(limit * 4, 20))
    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_limit,
        include=["documents", "metadatas", "distances"],
    )
    bm25, all_documents, all_metadatas, all_ids = build_bm25_index(collection)

    if bm25 is None:
        return []

    candidates: dict[str, dict] = {}
    vector_ids = (vector_results.get("ids") or [[]])[0]
    vector_documents = (vector_results.get("documents") or [[]])[0]
    vector_metadatas = (vector_results.get("metadatas") or [[]])[0]
    vector_distances = (vector_results.get("distances") or [[]])[0]
    space = (collection.metadata or {}).get("hnsw:space", "l2")

    for rank, (chunk_id, content, metadata, distance) in enumerate(
        zip(vector_ids, vector_documents, vector_metadatas, vector_distances),
        start=1,
    ):
        if not content:
            continue
        if space == "cosine":
            similarity = 1 - distance
        elif space == "ip":
            similarity = 1 - distance
        else:
            # Chroma L2 devuelve distancia euclídea al cuadrado; los vectores
            # E5 se normalizan, por lo que coseno equivale a 1 - distancia/2.
            similarity = 1 - distance / 2
        candidates[chunk_id] = {
            "content": content,
            "metadata": metadata or {},
            "similarity": similarity,
            "dense_rank": rank,
            "bm25_rank": None,
        }

    bm25_scores = bm25.get_scores(tokenize(question))
    lexical_indices = sorted(
        range(len(bm25_scores)),
        key=lambda index: bm25_scores[index],
        reverse=True,
    )[:candidate_limit]

    for rank, index in enumerate(lexical_indices, start=1):
        chunk_id = all_ids[index]
        candidate = candidates.setdefault(
            chunk_id,
            {
                "content": all_documents[index],
                "metadata": all_metadatas[index] or {},
                "similarity": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
            },
        )
        candidate["bm25_rank"] = rank

    # Reciprocal Rank Fusion combina la posición de cada fragmento en ambos
    # retrievers sin comparar directamente sus escalas de puntuación.
    rrf_k = 60
    for candidate in candidates.values():
        candidate["rank_score"] = sum(
            1 / (rrf_k + rank)
            for rank in (candidate["dense_rank"], candidate["bm25_rank"])
            if rank is not None
        )

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: candidate["rank_score"],
        reverse=True,
    )
    return [
        {
            "content": candidate["content"],
            "metadata": candidate["metadata"],
            "similarity": candidate["similarity"],
        }
        for candidate in ranked[:limit]
    ]
