# HUSC RAG 2025 — FastAPI backend
# Multi-stage: builder installs pinned deps, runtime is slim.
# Local layout (cwd=rag2025, PYTHONPATH=src) is mirrored: workdir /app holds
# src/, config/, prompts/, etc. — .dockerignore excludes data/, .venv, logs.

FROM python:3.13-slim AS builder

WORKDIR /app

# build-essential: needed to compile C extensions (lxml, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

# Core runtime deps only. Eval/arbiter libs (ragas, deepeval) are NOT baked
# in — see requirements-eval.txt; they are lazy-imported and absent from the
# running venv as of 2026-10-06.
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim

WORKDIR /app
ENV PYTHONPATH=src \
    PYTHONUNBUFFERED=1

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pre-seed the tiktoken BPE cache at build time so container start needs no network
# for tokenization. Lives in the runtime stage so the file persists in the final image.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Mirror the rag2025/ working layout (everything except what .dockerignore drops)
COPY src/ ./src/
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/

# Data (LanceDB + corpus) is a mounted volume, never baked into the image.
RUN mkdir -p /data
# Compat: graphrag_retriever.GRAPH_PATH resolves to /app/data/graph/...; point it at the mounted /data volume.
RUN ln -sfn /data /app/data
VOLUME ["/data"]

EXPOSE 8000

# Same invocation as local dev: cwd=/app, PYTHONPATH=src, src.main:app.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
