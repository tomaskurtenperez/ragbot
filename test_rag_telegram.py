"""Puente entre Telegram y el endpoint RAG de la API Ragbot."""

import logging
import os
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


load_dotenv()
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
RAGBOT_API_URL = os.getenv("RAGBOT_API_URL", "http://127.0.0.1:8000").rstrip("/")
TELEGRAM_MESSAGE_LIMIT = 4000


async def iniciar_api(application: Application) -> None:
    application.bot_data["ragbot_http"] = httpx.AsyncClient(
        base_url=RAGBOT_API_URL,
        timeout=httpx.Timeout(90.0, connect=10.0),
    )


async def cerrar_api(application: Application) -> None:
    client = application.bot_data.get("ragbot_http")
    if client:
        await client.aclose()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        _ensure_session(update, context)
        await update.message.reply_text(
            "Hola. Envíame una pregunta y Ragbot buscará en su base documental."
        )


def _ensure_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> str:
    chat_id = update.effective_chat.id if update.effective_chat else "unknown-chat"
    user_id = update.effective_user.id if update.effective_user else "unknown-user"
    sessions = context.user_data.setdefault("rag_sessions", {})
    chat_key = str(chat_id)
    if chat_key not in sessions:
        sessions[chat_key] = f"telegram:{chat_id}:{user_id}"
    return sessions[chat_key]


async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else "unknown-chat"
    user_id = update.effective_user.id if update.effective_user else "unknown-user"
    sessions = context.user_data.setdefault("rag_sessions", {})
    sessions[str(chat_id)] = f"telegram:{chat_id}:{user_id}:{uuid4().hex}"
    if update.message:
        await update.message.reply_text("Conversación nueva; empecemos de cero.")


async def responder(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    try:
        client: httpx.AsyncClient = context.application.bot_data["ragbot_http"]
        response = await client.post(
            "/chat",
            json={
                "question": update.message.text,
                "session_id": _ensure_session(update, context),
            },
        )
        response.raise_for_status()
        answer = response.json().get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("La API Ragbot devolvió una respuesta vacía o inválida")

        for start_index in range(0, len(answer), TELEGRAM_MESSAGE_LIMIT):
            await update.message.reply_text(
                answer[start_index : start_index + TELEGRAM_MESSAGE_LIMIT]
            )
    except httpx.HTTPStatusError as exc:
        logger.error(
            "Ragbot respondió HTTP %s: %s",
            exc.response.status_code,
            exc.response.text[:500],
        )
        await update.message.reply_text(
            "Ragbot no pudo responder la consulta. Revisa que la API esté activa."
        )
    except Exception:
        logger.exception("No se pudo consultar la API Ragbot")
        await update.message.reply_text(
            "No pude conectar con Ragbot. Comprueba que la API esté levantada."
        )


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("Configura TELEGRAM_BOT_TOKEN en el archivo .env.")

    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(iniciar_api)
        .post_shutdown(cerrar_api)
        .build()
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("new", new_chat))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, responder)
    )

    logger.info("Conectando Telegram con Ragbot en %s", RAGBOT_API_URL)
    application.run_polling()


if __name__ == "__main__":
    main()
