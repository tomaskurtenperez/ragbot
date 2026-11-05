from app.db.vector_db import search_similar


async def retrieve(
    question: str,
    top_k: int = 5
):

    return await search_similar(question=question, limit=top_k)
