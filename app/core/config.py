import os
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))
HF_EMBEDDING_MODEL = os.getenv(
    "HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
).strip()
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documents").strip()
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
LANGFUSE_BASE_URL = os.getenv(
    "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
).strip()
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DB_PATH = Path(
    os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "data" / "chroma_db"))
).resolve()
