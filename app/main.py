import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router
from app.core.config import LANGFUSE_ENABLED
from app.db.vector_db import warm_retriever
from app.services.embedding_service import get_embedding_model


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Preparando modelo de embeddings y retrievers antes de aceptar peticiones.")

    def warmup() -> None:
        get_embedding_model()
        warm_retriever()

    await asyncio.to_thread(warmup)
    logger.info("Modelo y retrievers listos.")
    yield
    if LANGFUSE_ENABLED:
        from langfuse import get_client

        get_client().flush()


app = FastAPI(
    title="Ragbot API",
    description=(
        "Retrieval-Augmented Generation API over IETF web and API standards "
        "(HTTP, OAuth 2.0, JWT, JSON). "
        "Hybrid retrieval (multilingual-e5 embeddings + BM25 fused with RRF) "
        "orchestrated with LangGraph and answered by Gemini."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
