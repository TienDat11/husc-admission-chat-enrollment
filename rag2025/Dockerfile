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
# CPU torch first (avoids multi-GB CUDA wheels); requirements then sees torch==2.11.0 satisfied.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0 torchvision==0.26.0 \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim
# libgomp1: runtime OpenMP lib required by the CPU torch wheel (not in python:slim).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=src \
    PYTHONUNBUFFERED=1

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Pre-seed the tiktoken BPE cache at build time so container start needs no network
# for tokenization. Lives in the runtime stage so the file persists in the final image.
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
RUN python -c "import torch, sentence_transformers; print('torch', torch.__version__)"

# Mirror the rag2025/ working layout (everything except what .dockerignore drops)
COPY src/ ./src/
COPY config/ ./config/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/
COPY alembic.ini ./
COPY alembic/ ./alembic/

# Data (LanceDB + corpus) is a mounted volume, never baked into the image.
RUN mkdir -p /data
# Compat: graphrag_retriever.GRAPH_PATH resolves to /app/data/graph/...; point it at the mounted /data volume.
RUN ln -sfn /data /app/data

EXPOSE 8000

# Boot: scripts/bootstrap_data.py extracts the data bundle into /data once (skipped when
# /data/lancedb exists), then exec uvicorn. The bundle is added by the deploy context.
CMD ["sh","-c","python scripts/bootstrap_data.py && exec uvicorn src.main:app --host 0.0.0.0 --port 8000"]
