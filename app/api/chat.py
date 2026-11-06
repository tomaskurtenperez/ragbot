from uuid import uuid4
import logging

from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest
from app.services.rag_service import generate_answer

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/chat")
async def chat(request: ChatRequest):
    session_id = (request.session_id or "").strip() or uuid4().hex
    logger.info(
        "Chat request received session_id=%s question=%r",
        session_id,
        request.question[:500],
    )
    try:
        answer = await generate_answer(request.question, session_id)
        return {"answer": answer, "session_id": session_id}
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error al consultar Gemini: {exc}") from exc
