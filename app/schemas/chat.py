from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
        description="Question about the indexed RFCs.",
        examples=["When should a server return 304 Not Modified?"],
    )
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Conversation id. Reuse it to keep follow-up context.",
    )


class ChatResponse(BaseModel):
    answer: str
    session_id: str
