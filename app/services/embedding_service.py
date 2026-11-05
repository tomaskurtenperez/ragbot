import asyncio
import os
from threading import Lock
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.core.config import HF_EMBEDDING_MODEL


_model: SentenceTransformer | None = None
_model_lock = Lock()
_encode_lock = Lock()


def _resolve_local_model() -> str:
    """Resolve the configured model to a local directory without Hub access."""
    configured_path = Path(HF_EMBEDDING_MODEL).expanduser()
    if configured_path.is_dir():
        return str(configured_path.resolve())

    hf_home = Path(
        os.getenv("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    ).expanduser()
    hub_cache = Path(os.getenv("HF_HUB_CACHE", str(hf_home / "hub"))).expanduser()
    repo_cache = hub_cache / f"models--{HF_EMBEDDING_MODEL.replace('/', '--')}"
    snapshots = repo_cache / "snapshots"

    if snapshots.is_dir():
        valid_snapshots = [
            snapshot
            for snapshot in snapshots.iterdir()
            if snapshot.is_dir()
            and (snapshot / "modules.json").is_file()
            and (snapshot / "config.json").is_file()
        ]
        if valid_snapshots:
            # El snapshot con modificación más reciente suele ser la revisión
            # que descargó Hugging Face más recientemente.
            return str(max(valid_snapshots, key=lambda path: path.stat().st_mtime))

    raise RuntimeError(
        f"No se encontró el modelo de embeddings '{HF_EMBEDDING_MODEL}' en "
        f"'{repo_cache}'. En Docker, monta el caché de Hugging Face del host "
        "en /root/.cache/huggingface."
    )


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                model_path = _resolve_local_model()
                _model = SentenceTransformer(
                    model_path,
                    local_files_only=True,
                )
    return _model


def generate_embeddings(
    texts: list[str],
    *,
    batch_size: int = 32,
):
    model = get_embedding_model()
    with _encode_lock:
        return model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


async def generate_query_embedding(question: str) -> list[float]:
    vectors = await asyncio.to_thread(
        generate_embeddings,
        [f"query: {question}"],
        batch_size=1,
    )
    return vectors[0].tolist()
