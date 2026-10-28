# RAG Pipeline Context Memo

Repo: `husc-admission-chat-enrollment`
Date: 2026-05-11

## Project Shape

- Offline build: `rag2025/scripts/build_graph.py` → chunk load → NER/triples → `data/graph/knowledge_graph.graphml` + `entity_index.json`.
- Core graph model: `rag2025/src/domain/graph.py` uses `NetworkX MultiDiGraph`, schema versioning, and PPR on a collapsed DiGraph view.
- Legacy online path: `rag2025/src/main.py` `/query` = HYDE multi-variant retrieval + optional hybrid dense/BM25 + rerank.
- Unified online path: `rag2025/src/main.py` `/v2/query` = auto-route between `padded_rag` and `graph_rag`.
- Graph path: `rag2025/src/services/query_router.py` + `rag2025/src/services/graphrag_retriever.py`.

## Key Components

- HYDE + step-back + route class: `rag2025/src/services/query_router.py`
- Dense LanceDB retrieval: `rag2025/src/services/lancedb_retrieval.py`
- Optional dense + BM25 fusion: `rag2025/src/services/hybrid_search.py`
- Query embedding: `rag2025/src/services/embedding.py`
- Rerank: `rag2025/src/services/reranker.py`
- Graph build: `rag2025/scripts/build_graph.py`
- Graph PPR fusion + incremental graph update: `rag2025/src/services/graphrag_retriever.py`
- Notebook eval helpers: `packages/rag-chatbot-husc/src/notebooks/eval_core.py`

## Assessment

- Graph + PaddedRAG + HYDE is sound for HUSC admissions.
- What is sound:
  - simple fact queries stay on dense retrieval
  - multi-hop / comparative queries can use graph PPR
  - HYDE + step-back helps Vietnamese query expansion
  - graph is persisted, versioned, and incrementally updatable
- What is risky:
  - graph build still depends on LLM-based NER extraction
  - `/v2/query` baseline is dense LanceDB only; BM25 hybrid exists in legacy `/query` when enabled, but not in the unified path
  - the notebook eval is still Colab-shaped
- Keep:
  - `padded_rag` vs `graph_rag` split
  - PPR fusion
  - incremental graph updates
- Change lightly:
  - add deterministic NER fallback for graph build
  - decide whether to port BM25 hybrid into unified `/v2/query`
  - tune router threshold and PPR weights from local eval

## BGE Recommendation

Default: `BAAI/bge-m3`

- Official BAAI multilingual model.
- Specs on model card: 1024-dim, 8192-token max length, dense + sparse + multi-vector retrieval.
- Good fit for mixed Vietnamese/English retrieval.
- Better local fit than `BAAI/bge-multilingual-gemma2`, which is LLM-based and backed by `gemma-2-9b`.

Tradeoff:

- `bge-m3` is the practical low-disk choice.
- `bge-multilingual-gemma2` is heavier and only worth it if disk/GPU budget is loose and you need the larger model family.
- Smaller English-only BGE variants are not a fit for this corpus.

## Notebook Eval

Old notebook: `packages/rag-chatbot-husc/src/notebooks/colab_eval_v5_thesis.ipynb`
Current notebook routing target: `deepseek-v4-flash` for graph/HYDE, `gpt-5.4` for final answers.

Local helper: `packages/rag-chatbot-husc/src/notebooks/eval_core.py`

Test set:

- primary: `rag2025/results/test_questions.json`
- fallback: `rag2025/results/test_questions.json`
- current set size: 50
- no 90-question cell found in repo copy; notebook loads all rows from `test_questions.json`
- split: 20 `simple`, 20 `multihop`, 10 `comparative`

Compare:

- `padded_rag`
- `graph_rag`
- auto routing

Enough metrics:

- semantic accuracy
- retrieval recall proxy
- hallucination rate
- groundedness
- p95 latency
- route split / route correctness
- zero-error run requirement

## Local Eval Path

1. Load `rag2025/results/test_questions.json`.
2. Call `/v2/query` with `force_route=padded_rag`, `force_route=graph_rag`, and auto.
3. Use `eval_core.py` to normalize output and score:
   - `semantic_correctness`
   - `retrieval_recall_proxy`
   - `hallucination_flag`
   - `latency_stats`
   - `compute_route_parity`
4. Prefer one embedding only for low disk, ideally `BAAI/bge-m3`.
5. Keep only zero-error runs when choosing config.

## Lightweight Upgrades

Priority 1:

- keep `BAAI/bge-m3` as default embedding
- add a local eval script around `eval_core.py`
- decide whether unified `/v2/query` should get BM25 fusion

Priority 2:

- add deterministic NER fallback for graph build
- prune unused embedding artifacts
- log route correctness and route mismatch

Priority 3:

- tune `GRAPHRAG_SIMPLE_THRESHOLD`
- tune `GRAPHRAG_PPR_ALPHA`
- tune hybrid fusion weights if BM25 is ported into `/v2/query`

## Next Session Notes

- Start from `rag2025/src/main.py`, `rag2025/src/services/graphrag_retriever.py`, and `packages/rag-chatbot-husc/src/notebooks/eval_core.py`.
- Use only one embedding model locally.
- Keep disk footprint low.
- Current local cache target is D: via `HF_HOME=D:/chunking/.cache/huggingface`.
- Current role model split is graph/HYDE `deepseek-v4-flash`, final `gpt-5.4`.
- If BGE docs are needed again:
  - `https://huggingface.co/BAAI/bge-m3`
  - `https://huggingface.co/BAAI/bge-multilingual-gemma2`
