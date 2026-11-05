from google import genai
from google.genai import types
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL


def _format_source(chunk: dict) -> str:
    metadata = chunk.get("metadata", {})
    label = metadata.get("filename", "desconocida")
    if metadata.get("page"):
        label += f", página {metadata['page']}"
    return f"[Fuente: {label}]"


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

    prompt = f"""You are an assistant specializing in European Union documentation, policies, regulations, and initiatives.

Your role is to respond to user inquiries in a clear, precise, formal, and natural manner, using the provided documentary context as your primary source.

Instructions:

1. Always respond in English, maintaining a professional, objective, and direct tone.
2. Begin directly with the answer to the inquiry. Do not include greetings, introductions, generic opening statements, or phrases such as "Dear user," "In response to your inquiry," or "The following details...".
3. Do not unnecessarily repeat or rephrase the user's question.
4. Do not include closings, expressions of gratitude, offers of further assistance, or signatures at the end of your response.
5. Organize information logically. Use paragraphs for general explanations, and numbered or bulleted lists when they help present multiple points more clearly.
6. Use subheadings only when the length or complexity of the response warrants them.
7. Base your responses exclusively on the provided documentary context. Do not invent data, dates, figures, references, or conclusions.
8. If the available information is insufficient to provide a complete answer, state this briefly and address only those aspects supported by the documents.
9. If you cannot find relevant information to answer the inquiry, state this briefly and directly, without attempting to supplement the response with external knowledge.
10. Retain the official names of European Union institutions, documents, programs, regulations, and bodies.
11. Do not include references to the specific documents or pages from which the information is drawn. Do not invent bibliographic references or page numbers.
12. Avoid long direct quotations. Prioritize explaining the information in your own words, while preserving its original meaning. 13. Do not use unnecessary phrases such as "based exclusively on the context," "according to the information provided," or "as a language model," unless it is necessary to clarify a specific limitation.

Recent conversation (use only to understand follow-up references; facts must be grounded in the document context):
{conversation}

Document context:
{context}

Query:
{question}

Response:"""

    async with genai.Client(api_key=GEMINI_API_KEY).aio as client:
        response = await client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
            max_output_tokens=150
        )
        )

    answer = response.text
    if answer:
        return answer.strip()

    return "Gemini no devolvió texto."
