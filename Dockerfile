FROM python:3.12-slim AS build

WORKDIR /app
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONPATH=/app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data/documents ./data/documents

# Download the same embedding model used at runtime, then create the Chroma
# index from the PDFs included in the repository.
RUN python -c "from huggingface_hub import snapshot_download; from app.core.config import HF_EMBEDDING_MODEL; snapshot_download(repo_id=HF_EMBEDDING_MODEL)" \
    && python -m scripts.ingest

FROM python:3.12-slim AS runtime

WORKDIR /app
ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONPATH=/app
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

COPY --from=build /usr/local /usr/local
COPY app ./app
COPY --from=build /app/data/chroma_db ./data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
