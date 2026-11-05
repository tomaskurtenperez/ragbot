from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="Pregunta para Gemini")
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
