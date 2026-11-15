import logging
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import generate_answer

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(request: ChatRequest) -> ChatResponse:
    session_id = (request.session_id or "").strip() or uuid4().hex
    logger.info(
        "Chat request received session_id=%s question=%r",
        session_id,
        request.question[:500],
    )
    try:
        answer = await generate_answer(request.question, session_id)
    except RuntimeError as exc:
        # Configuration problems (e.g. missing API key) are safe to surface.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Answer generation failed session_id=%s", session_id)
        raise HTTPException(
            status_code=502,
            detail="The language model could not generate an answer. Try again later.",
        ) from exc
    return ChatResponse(answer=answer, session_id=session_id)
