# RAG2025 – Project Rules & SOP

## 1. Build & Run Commands

```bash
# Activate virtual environment (Windows)
venv\Scripts\activate.bat

# Install / update dependencies
pip install -r requirements.txt

# Run API server (production launcher with preflight)
run_lab.bat

# Run API server (manual, no preflight)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# Swagger UI
http://localhost:8000/docs
```

## 2. Test & Validation Commands

```bash
# Preflight check (LanceDB + config + data)
python scripts/preflight_check.py

# Verify LanceDB table health
python scripts/check_lancedb.py

# Run test retriever
python test_retriever.py

# Run Qdrant connectivity test
python test_qdrant.py

# Verify API endpoints
python scripts/verify_api.py
```

## 3. Data Pipeline Commands

```bash
# Ingest all JSONL chunks into LanceDB
python scripts/ingest_lancedb.py

# Ingest HUSC-specific chunks
python scripts/ingest_husc_chunks.py

# Enhance chunks with metadata
python scripts/enhance_chunks.py

# Build local FAISS/NumPy index (fallback)
python scripts/build_index.py

# Migrate from legacy store to Qdrant
python scripts/migrate_to_qdrant.py
```

## 4. GraphRAG Commands (NEW – Additive Layer)

```bash
# Build knowledge graph from chunked JSONL files
python scripts/build_graph.py

# Verify graph output
python scripts/verify_graph.py

# GraphRAG evaluation + ablation templates
python scripts/evaluate_graphrag.py
python scripts/build_ablation_template.py
python scripts/build_eval_dataset_template.py
```

## 5. LanceDB Rules

| Rule | Detail |
|------|--------|
| ✅ MANDATORY | Use embedded LanceDB as the primary dense store |
| Collection (chunks) | `rag2025` – vector size 4096, Cosine-style dense retrieval |
| Collection (entities) | `husc_entities` – vector size 4096 |

## 6. Coding Standards

- **Language**: Python 3.10+
- **Style**: DDD (Domain-Driven Design) – Domain layer > Application layer > Infrastructure layer
- **Formatting**: `black` (configured in requirements.txt)
- **Logging**: `loguru` only – no `print()` in production code
- **Retry logic**: `tenacity` with exponential backoff
- **Type hints**: Mandatory on all public methods
- **Docstrings**: Google style on all public classes/methods
- **No comments** on obvious code; only comment WHY, not WHAT

## 7. LLM Model Priority

| Task | Primary | Fallback |
|------|---------|---------|
| HyDE / Step-Back / Routing | gemini-2.5-flash via `ramclouds.me/v1` | Groq (Llama 3.1) |
| NER / Triple extraction | gemini-2.5-flash via `ramclouds.me/v1` | Groq (Llama 3.1) |
| RAG generation | gemini-2.5-flash via `ramclouds.me/v1` | Groq |
| Evaluation / judging | gemini-2.5-flash (temperature=0.1) | — |

**Config**: Set `RAMCLOUDS_API_KEY` in `.env`. See `.env.example` for full template.
`UnifiedLLMClient` in `src/services/llm_client.py` handles all provider fallback automatically.

## 8. Directory Layout

```
rag2025/
├── src/
│   ├── domain/               # NEW – DDD Domain layer
│   │   ├── entities.py       # Domain entity models
│   │   └── graph.py          # Graph domain logic
│   ├── services/             # Application services
│   │   ├── embedding.py
│   │   ├── retriever.py
│   │   ├── lancedb_retrieval.py
│   │   ├── query_enhancer.py
│   │   ├── llm_generator.py
│   │   └── graphrag_retriever.py  # NEW
│   └── main.py
├── scripts/
│   ├── build_graph.py        # NEW – offline graph builder
│   ├── evaluate_graphrag.py  # NEW – evaluation pipeline
│   └── ...existing scripts...
├── data/
│   ├── chunked/              # 11 JSONL files (input)
│   └── graph/                # NEW – graph artifacts
│       ├── knowledge_graph.graphml
│       └── entity_index.json
├── rules/                    # THIS FOLDER – SOPs
└── plan/                     # Implementation plans
```

## 9. Git Rules

- Branch: `feature/<issue>-<desc>` or `fix/<issue>-<desc>`
- Commit: `feat(#N): description` / `fix(#N): description`
- Never: `git add -A`, `git add .`
- Never commit: `.env`, `venv/`, `data/`, `*.graphml` (large artifacts)

## 10. Environment Variables

```bash
GEMINI_API_KEY=...       # Gemini 2.5 Flash (primary)
GROQ_API_KEY=...         # Groq Llama (HyDE primary)
OPENAI_API_KEY=...       # Z.AI/GLM-4 endpoint
LANCEDB_URI=...           # Local LanceDB URI
LANCEDB_TABLE=rag2025
LANCEDB_ENTITY_TABLE=husc_entities
QWEN_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
QWEN_EMBEDDING_DIM=4096
```
