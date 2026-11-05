import asyncpg
from app.core.config import settings

pool = None

async def connect_db():
    global pool

    pool = await asyncpg.create_pool(
        settings.DATABASE_URL
    )


async def close_db():
    await pool.close()


async def search_similar(
    embedding: list[float],
    limit: int = 5
):

    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

    query = """
        SELECT
            id,
            content,
            metadata,
            1 - (embedding <=> $1::vector) AS similarity
        FROM documents
        ORDER BY embedding <=> $1::vector
        LIMIT $2;
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            query,
            embedding_str,
            limit
        )

    return [dict(row) for row in rows]