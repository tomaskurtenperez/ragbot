from google import genai
from google.genai import types
from app.core.config import GEMINI_API_KEY, GEMINI_MAX_OUTPUT_TOKENS, GEMINI_MODEL
from app.core.prompts import RAG_PROMPT_TEMPLATE


def _format_source(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    label = metadata.get("filename", "unknown")
    if metadata.get("section"):
        label += f", section {metadata['section']}"
    elif metadata.get("page"):
        label += f", page {metadata['page']}"
    return f"[Source: {label}]"


async def generate_response(
    question: str,
    chunks: list[dict],
    history: list[dict[str, str]] | None = None,
) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "Falta configurar GEMINI_API_KEY en el entorno o en el archivo .env"
        )

    context = "\n\n".join(
        f"{_format_source(chunk)}\n{chunk['content']}"
        for chunk in chunks
        if chunk.get("content")
    )
    if not context:
        context = "No relevant fragments were found in the document database."

    conversation = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in (history or [])
        if message.get("role") in {"user", "assistant"} and message.get("content")
    )
    if not conversation:
        conversation = "No previous messages."

    prompt = RAG_PROMPT_TEMPLATE.format(
        conversation=conversation,
        context=context,
        question=question,
    )

    async with genai.Client(api_key=GEMINI_API_KEY).aio as client:
        response = await client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
            ),
        )

    answer = response.text
    if answer:
        return answer.strip()

    return "Gemini no devolvió texto."
