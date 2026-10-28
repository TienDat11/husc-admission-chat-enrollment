"""Ticket 29 — chaos publish/rollback drill.

Bombs the REAL ``/v2/query`` handler of ``rag2025/src/main.py`` with concurrent
requests while firing a real ``POST /admin/reload-table`` (publish) and a real
``POST /admin/rollback`` at scheduled offsets, then scans every response for
chunks carrying two different ``release_id`` values.

Design: ``.scratch/husc-rag-admin-upgrade/ticket29-chaos-drill-design.md``.
Follows §7 (no lifespan; importlib + ``configure_canonical_record_store`` by
hand), §1 (release_id in ``metadata_json`` AND the top-level column), §6 (second
table ``husc_chunks_next`` so reload is a real publish), §8 (limiter bypassed,
enumeration query so the router short-circuits with zero LLM calls), §9 (query
cache disabled AND query text nonce'd), §10 (hybrid OFF), §11 (rollback targets
the live serving table).

CAVEAT THAT TRAVELS WITH EVERY NUMBER THIS SCRIPT PRINTS:
  the LLM provider is DOWN in this environment (ramclouds 403
  insufficient_user_quota, groq 401 invalid_api_key), so the GENERATOR is
  stubbed. Latencies reported here are retrieval + routing + fusion + payload
  build ONLY. They are NOT end-to-end LLM latency.

Usage
    python scripts/chaos_publish_rollback.py --scenario scenario.json --out DIR
    python scripts/chaos_publish_rollback.py --mutation-proof   # detector bites

Scenario JSON keys: workers, duration_s, rate_rps, publish_at_s,
rollback_at_s, query_delay_s, seed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths / env — MUST be set before src/main.py is imported (its module-level
# ``settings = RAGSettings()`` reads the environment and the .env file).
# ---------------------------------------------------------------------------
RAG2025 = Path(__file__).resolve().parents[1]
REPO_ROOT = RAG2025.parent
for _p in (str(REPO_ROOT), str(RAG2025), str(RAG2025 / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.chdir(RAG2025)

TABLE_A = "husc_chunks"
TABLE_B = "husc_chunks_next"
RELEASE_A = "REL29A"
RELEASE_B = "REL29B"
ENUM_QUERY = "HUSC có những ngành đào tạo nào?"
VECTOR_DIM = 640  # smallest value the EMBEDDING_DIM validator accepts

#: The `top_k` `_one_query` sends. `/v2/query` asks the retriever for
#: `top_k * 3` rows (`main.py:1586`, `:1610`), so this fixes the baseline window.
REQUEST_TOP_K = 5
BASELINE_WINDOW = REQUEST_TOP_K * 3

#: Rows whose vector IS the query anchor, i.e. distance 0, so they always
#: occupy the whole baseline window and push every booster id OUT of it.
#:
#: This constant is load-bearing and MUST exceed ``BASELINE_WINDOW``. The
#: booster only calls ``fetch_by_id`` for chunk ids ABSENT from the baseline
#: (`aggregation_booster.py:110-131`), and that second read is the ONLY
#: intra-request read that can observe two releases. Without these rows the
#: seeded corpus (3 booster + 8 extra = 11) fits entirely inside the 15-row
#: window, ``seen_ids`` swallows all three booster ids, ``fetch_by_id`` is never
#: called, and a "0 mixed-release answers" verdict is vacuous — it reports the
#: absence of a read that never happened.
FILLER_COUNT = 20
# A real-looking model tag, deliberately NOT a `Fallback(...)` prefix:
# `is_generation_failure()` (llm_generator.py:341) treats that prefix as a
# terminal no-model-output tag, which would mark every drill answer degraded.
STUB_PROVIDER = "chaos-drill-stub (no-llm)"

# The two credentials the drill authenticates with. They are distinct because
# the grants are: `reload-table` needs ops.reload_table and `rollback` needs
# records.rollback, which the ops role does not hold.
OPS_TOKEN = "chaos-drill-ops-token"
ADMIN_TOKEN = "chaos-drill-admin-token"

DEFAULT_SCENARIO: Dict[str, Any] = {
    "workers": 8,
    "duration_s": 12.0,
    "rate_rps": 10.0,
    "publish_at_s": 4.0,
    "rollback_at_s": 8.0,
    "query_delay_s": 0.0,
    "seed": 29,
}


def _load_dotenv() -> Dict[str, str]:
    """Parse rag2025/.env so the drill reuses the REAL config values."""
    out: Dict[str, str] = {}
    env_file = RAG2025 / ".env"
    if not env_file.exists():
        return out
    for line in env_file.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


DOTENV = _load_dotenv()
CURRENT_ADMISSION_YEAR = int(DOTENV.get("CURRENT_ADMISSION_YEAR", "2026"))

# ---------------------------------------------------------------------------
# Stand-ins (design §7: stub ONLY the generator; everything else is real)
# ---------------------------------------------------------------------------


class FakeEmbedding:
    """Deterministic local embedding: blake2b(query|slot) -> unit vector.

    NOT the production Qwen3-Embedding vectors: the provider/model weights are
    not exercised by this drill. Vector values are irrelevant to the
    mixed-release question — LanceDB returns top-k rows for the query point, and
    the drill reads the release_id those rows carry.

    ``encode_query`` is pinned to a CONSTANT anchor vector (it does NOT hash the
    query text). That is deliberate and load-bearing: the booster's second read
    (``fetch_by_id``) is the only intra-request read that can observe two
    releases, and it only runs for chunk ids ABSENT from the baseline. A
    text-dependent query vector would rank the seeded rows arbitrarily, so the
    booster would sometimes be skipped and the drill would silently stop
    testing the seam. Pinning the query vector lets the seeded filler rows sit
    at distance 0 (always ranked first), which pushes every booster id out of
    the baseline on EVERY request — see ``FILLER_COUNT``.
    """

    #: Text whose vector is the constant query anchor (see class docstring).
    QUERY_VECTOR_TEXT = "__chaos_drill_query_anchor__"

    def __init__(self, dim: int = VECTOR_DIM) -> None:
        self.embedding_dim = dim
        self.model_name = "chaos-drill-deterministic-blake2b"
        self.device = "cpu"

    def _vec(self, text: str) -> np.ndarray:
        buf = bytearray()
        counter = 0
        while len(buf) < self.embedding_dim * 2:
            buf += hashlib.blake2b(
                f"{text}#{counter}".encode("utf-8"), digest_size=64
            ).digest()
            counter += 1
        raw = np.frombuffer(bytes(buf[: self.embedding_dim * 2]), dtype=np.uint16)
        vec = raw.astype(np.float32) - 32768.0
        norm = float(np.linalg.norm(vec))
        return vec / (norm if norm else 1.0)

    def anchor(self) -> np.ndarray:
        """The constant query vector itself.

        A row stored with THIS vector sits at distance 0 from every query, so it
        always ranks inside the baseline window — that is how the filler rows
        keep the booster ids out of the dense read.
        """
        return self._vec(self.QUERY_VECTOR_TEXT)

    def encode_query(self, text: str) -> np.ndarray:
        # Deliberately text-INdependent — see class docstring. Every query maps
        # to the same anchor point so the seeded filler rows (distance 0) always
        # outrank the booster rows, keeping `fetch_by_id` reachable.
        return self.anchor()

    def encode_documents(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def warmup(self) -> None:  # pragma: no cover - parity hook
        return None


class StubLLM:
    """Proves the router's LLM call is never reached: any call raises."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise RuntimeError("chaos-drill: router LLM must not be called")

    async def chat_json(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        raise RuntimeError("chaos-drill: router LLM must not be called")

    async def generate(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        raise RuntimeError("chaos-drill: router LLM must not be called")


class StubNER:
    """Deterministic, LLM-free NER stand-in (design §7).

    The enum fast-path still runs GraphRAG fusion, which seeds PPR from
    ``_ner.extract_from_query`` (`graphrag_retriever.py:195-218`). Returning no
    seeds keeps ``ppr_scores`` empty, so the expander contributes nothing and
    the served answer is built from the real LanceDB rows only. ``extract``
    never runs (it is the no-seeds fallback) but is provided so a change in that
    path fails loudly here instead of reaching a dead provider.
    """

    def __init__(self) -> None:
        self.query_calls = 0
        self.chunk_calls = 0

    def extract_from_query(self, query: str) -> List[str]:
        self.query_calls += 1
        return []

    async def extract(self, chunk: Any) -> Any:
        from src.domain.entities import ExtractionResult

        self.chunk_calls += 1
        return ExtractionResult(
            chunk_id=getattr(chunk, "chunk_id", "__query__"),
            entities=[],
            triples=[],
            error="chaos-drill: NER is stubbed",
        )


async def _no_rate_limit(request: Any) -> None:
    """Async no-op — the real `_enforce_rate_limit` is awaited at every entry
    point (`main.py:1472`, `:2163`), so a sync lambda would raise TypeError."""
    return None


class StubGenerator:
    """The ONLY stub in this drill (design §7): app, routes, tracker, drain
    gate, retriever, booster, router and fusion are all the real ones.

    Echoes the chunks it was served verbatim, one line per chunk, so the
    mixed-release detector reads the release_id the REAL retrieval path
    attached to each document.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def generate_answer(self, query, chunks, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        self.calls += 1
        answer = _synth_answer(query, chunks, kwargs.get("history"))
        return {
            "answer": answer,
            "sources": [],
            "confidence": float(kwargs.get("confidence") or 0.0),
            "provider": STUB_PROVIDER,
            "chunks_used": len(chunks),
            "generation_failed": False,
        }

    async def generate_answer_stream(self, query, chunks, *args: Any, **kwargs: Any):
        """Yields raw token pieces — the real stream contract
        (`llm_generator.py:1016+` yields `str`, and the caller accumulates)."""
        self.calls += 1
        yield _synth_answer(query, chunks, kwargs.get("history"))


def _synth_answer(query: str, chunks: List[Dict[str, Any]], history: Any = None) -> str:
    """Echo the injected aggregation chunks verbatim (they ARE the list answer).

    ``chunks`` is exactly what ``main._build_v2_chunks`` built, so the
    ``metadata`` each dict carries is the real ``RetrievedDocument.metadata``
    from the real retrieval path (design §1).
    """
    del history
    lines = [f"[chaos-drill stubbed generation] query={query!r}"]
    for ch in chunks:
        meta = ch.get("metadata") or {}
        lines.append(
            " | ".join(
                (
                    str(meta.get("release_id", "?")),
                    str(ch.get("chunk_id")),
                    str(ch.get("score")),
                    str(ch.get("text", ""))[:90],
                )
            )
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Row seeding
# ---------------------------------------------------------------------------


def make_row(
    chunk_id: str,
    release_id: str,
    text: str,
    embedding: FakeEmbedding,
    anchor: bool = False,
) -> Dict[str, Any]:
    """A LanceDB row in the shape the served table uses.

    ``metadata_json`` is a JSON STRING carrying release_id (design §1: without
    that copy the release_id never reaches RetrievedDocument.metadata, so the
    detector could not see it). The top-level ``release_id`` column is also
    kept: the publish step's delete predicate filters on it.
    """
    vector = embedding.anchor() if anchor else embedding._vec(text)
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "text": text,
        "vector": vector.tolist(),
        "metadata_json": json.dumps(
            {"release_id": release_id, "data_year": CURRENT_ADMISSION_YEAR}
        ),
        "source": "husc",
        "release_id": release_id,
    }


def build_release_rows(
    ids: List[str], extra_ids: List[str], release_id: str, embedding: FakeEmbedding
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    # Filler rows FIRST, at distance 0, so the baseline window is saturated
    # before the booster rows are ranked — see ``FILLER_COUNT`` for why the
    # drill is vacuous without them.
    for i in range(FILLER_COUNT):
        rows.append(
            make_row(
                f"chaos_t29_{CURRENT_ADMISSION_YEAR}_filler_{i:02d}",
                release_id,
                f"FILLER {release_id} dong {i:02d} — du lieu nen {CURRENT_ADMISSION_YEAR}",
                embedding,
                anchor=True,
            )
        )
    for cid in ids:
        rows.append(
            make_row(
                cid,
                release_id,
                f"DANH MUC {release_id}: {cid.replace('_', ' ')} — danh mục tuyển sinh {CURRENT_ADMISSION_YEAR} của {release_id}",
                embedding,
            )
        )
    for cid in extra_ids:
        rows.append(
            make_row(
                cid,
                release_id,
                f" Noi dung {cid} cua {release_id} nganh {CURRENT_ADMISSION_YEAR}",
                embedding,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Mixed-release detector
# ---------------------------------------------------------------------------


def detect_mixed_release(answer: str) -> List[str]:
    """Release ids observed in one generated answer.

    The stub generator prints ``<release_id> | <chunk_id> | <score> | <text>``
    per chunk. The release id is the first field of the line, so it is counted
    once per chunk and cannot be double-counted from the free-text tail.
    """
    releases = set()
    for line in (answer or "").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 2 and parts[0].startswith(RELEASE_PREFIXES):
            releases.add(parts[0])
    return sorted(releases)


RELEASE_PREFIXES = (RELEASE_A, RELEASE_B)


# ---------------------------------------------------------------------------
# Scenario result container
# ---------------------------------------------------------------------------


@dataclass
class DrillResult:
    scenario: Dict[str, Any]
    requests: List[Dict[str, Any]] = field(default_factory=list)
    injections: List[Dict[str, Any]] = field(default_factory=list)
    errors_by_status: Dict[str, int] = field(default_factory=dict)
    mixed: List[Dict[str, Any]] = field(default_factory=list)
    latency: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    seed_ids: List[str] = field(default_factory=list)
    extra_ids: List[str] = field(default_factory=list)
    booster_ids: List[str] = field(default_factory=list)
    vector_dim: int = VECTOR_DIM
    #: True for the mutation proof, which must OBSERVE a mixed answer. The drill
    #: must observe NONE. Both render through :func:`verdict_line` so the printed
    #: verdict, the artifact's ``verdict`` field and the exit code cannot drift.
    expect_mixed: bool = False
    caveat: str = (
        "Latencies are NOT end-to-end LLM latency. The LLM provider is down "
        "(ramclouds 403 insufficient_user_quota, groq 401 invalid_api_key) and "
        "the generator is stubbed; figures cover routing + retrieval + fusion + "
        "payload build only. Embeddings are a deterministic local blake2b "
        "stand-in, not the production Qwen3-Embedding vectors."
    )


def _pct(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _latency_stats(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    ok = [r["latency_ms"] for r in rows if r["status"] == 200]
    bad = [r["latency_ms"] for r in rows if r["status"] != 200]
    return {
        "n_200": len(ok),
        "n_non200": len(bad),
        "p50_ms": _pct(ok, 0.50),
        "p95_ms": _pct(ok, 0.95),
        "max_ms": max(ok) if ok else None,
    }


# ---------------------------------------------------------------------------
# App construction (design §7)
# ---------------------------------------------------------------------------


def _load_main():
    spec = importlib.util.spec_from_file_location(
        "chaos_drill_main", str(RAG2025 / "src" / "main.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(RAG2025 / "src"))
    spec.loader.exec_module(module)
    return module


@asynccontextmanager
async def _no_lifespan(app: Any):
    """Design §7: the drill runs the app with NO startup.

    ``src/main.py``'s real lifespan declares ``global`` for all nine services
    and rebuilds every one of them — embedding (real Qwen), retriever, pipeline,
    reranker, generator, query cache, guardrail — so entering a ``TestClient``
    over the unmodified app silently discards every stub this script installs.
    Observed symptom: ``Model dimension mismatch: expected 640, got 1024`` (real
    Qwen vectors against the drill's 640-d table), retrieval dead, and the real
    LLM generator in the answer path (``provider=Fallback(Error)``).
    """
    yield None


def _neutralize_lifespan(main: Any) -> None:
    """Stop the app's lifespan from clobbering the stubs — BEFORE the client opens.

    ``TestClient`` must still be ENTERED, not used bare: ``__enter__`` starts one
    blocking portal that every request shares (``starlette/testclient.py:414-418``
    reuses ``self.portal``), which is what keeps ``RequestTracker``'s asyncio
    events on ONE event loop. A bare ``TestClient(app)`` starts a fresh loop per
    request, so the reload's ``resume()`` cannot wake the blocked queries and the
    drain proof measures the timeout instead of the drain —
    ``tests/integration/test_reload_under_load.py:203-207`` documents exactly
    that harness bug. So: keep the entered client, replace the lifespan.
    """
    main.app.router.lifespan_context = _no_lifespan

def _wire_app_state(main: Any, embedding: Any, reranker: Any = None) -> None:
    """Set the ``app.state`` a real startup would have left behind.

    Paired with :func:`_neutralize_lifespan`: the lifespan is the only writer of
    these, so without this the app is left mid-startup. ``warmup_stage`` is read
    by ``routers/health.py:88``; the rest are written by the lifespan at
    ``main.py:345-347`` and are set here so the drill's app state matches a real
    boot rather than a partially-initialised one.
    """
    main.app.state.warmup_stage = "ready"
    main.app.state.embedding_service = embedding
    main.app.state.reranker_service = reranker
    main.app.state._metrics_snapshot = main._metrics_snapshot

@contextmanager
def _count_fetch_by_id() -> Iterator[Dict[str, int]]:
    """Count ``LanceDBRetriever.fetch_by_id`` calls, as evidence the seam ran.

    Patched on the CLASS, not on an instance: ``/admin/reload-table`` and
    ``/admin/rollback`` rebind the serving retriever to a NEW object through
    ``swap_runtime_services``, so an instance-level wrapper would stop counting
    the moment the swap this drill exists to probe actually happens.

    Without this counter a "0 mixed-release answers" verdict is unfalsifiable:
    the booster's second read is the only one that can observe two releases, and
    a run where it is never called looks identical to a run where it was called
    and stayed consistent.
    """
    from src.services.lancedb_retrieval import LanceDBRetriever

    box = {"n": 0}
    original = LanceDBRetriever.fetch_by_id

    def counting(self: Any, chunk_id: str) -> Any:
        box["n"] += 1
        return original(self, chunk_id)

    LanceDBRetriever.fetch_by_id = counting
    try:
        yield box
    finally:
        LanceDBRetriever.fetch_by_id = original

def _configure_env(workdir: Path) -> Dict[str, str]:
    """Point the app at a throwaway LanceDB dir + store; keep the real config."""
    overrides = {
        "APP_ENV": "dev",
        "LOG_LEVEL": "WARNING",
        "LANCEDB_URI": str(workdir / "lancedb"),
        "LANCEDB_TABLE": TABLE_A,
        "EMBEDDING_DIM": str(VECTOR_DIM),
        "CANONICAL_STORE_ROOT": str(workdir / "store"),
        "USE_HYBRID_RETRIEVAL": "false",  # design §10
        "RATE_LIMIT_PER_MINUTE": "100000",  # design §8 (belt)
        "RERANKER_ENABLED": DOTENV.get("RERANKER_ENABLED", "true"),
        "RERANKER_ADAPTIVE_CONFIDENCE": DOTENV.get(
            "RERANKER_ADAPTIVE_CONFIDENCE", "0.5"
        ),
        "CURRENT_ADMISSION_YEAR": str(CURRENT_ADMISSION_YEAR),
        "RAG_CONF_THRESHOLD": DOTENV.get("RAG_CONF_THRESHOLD", "0.35"),
    }
    os.environ.update(overrides)
    return overrides


def _seed_lancedb(workdir: Path, rows_a: List[dict], rows_b: List[dict]) -> None:
    """Create both tables with the release_id INSIDE ``metadata_json``.

    Design §1: ``_normalize_rows`` parses the JSON string into
    ``RetrievedDocument.metadata``, which is what ``_build_v2_chunks`` carries
    to the generator. The top-level column is only for the delete predicate.

    TABLE_B is given a real publisher history, not just its own rows: at v0 it
    holds release A's rows tagged ``release-REL29A`` — the same state a real
    publish leaves on the table it builds — and only then does release B's
    content replace A's and get tagged. Without that v0, the rollback's
    ``LanceDBRetrievalRollback.validate`` reads the tags of the LIVE table
    (post-publish that is TABLE_B), does not find ``release-REL29A``, and the
    injection 409s with TargetNotTaggedError instead of rolling back.
    """
    import lancedb

    from src.infrastructure.lancedb_versioning import tag_release

    db = lancedb.connect(str(workdir / "lancedb"))
    # The serving table at seed time: release A's rows, tagged.
    tbl_a = db.create_table(TABLE_A, data=rows_a)
    tag_release(tbl_a, RELEASE_A)
    # The next table: A's rows tagged REL29A (the publish-time starting state),
    # then A's rows deleted and B's rows added, tagged REL29B.
    tbl_b = db.create_table(TABLE_B, data=rows_a)
    tag_release(tbl_b, RELEASE_A)
    tbl_b.delete(f"release_id = '{RELEASE_A}'")
    tbl_b.add(rows_b)
    tag_release(tbl_b, RELEASE_B)


def _seed_store(workdir: Path, main: Any, settings: Any, chunk_count: int = 0) -> None:
    """Release manifests + active pointer, in the layout the readers expect.

    ``write_manifest(release_dir, manifest)`` writes
    ``<release_dir>/release_manifest.json`` and ``AdminRollbackService`` reads a
    release back through ``_release_dir(release_id)`` = ``<releases>/<id>/``
    (`src/services/admin_rollback.py:939-942`), so the argument is a DIRECTORY
    named after the release — passing ``<id>.json`` would create a directory
    literally named ``REL29A.json`` and every rollback would 404.

    The field set is the real :class:`ReleaseManifest` (`src/domain/release.py`):
    ``version``/``source_batch``/``row_count``/``created_by`` do not exist on it.
    ``lancedb_tag`` is set with the same helper the rollback path uses, because
    that is the tag a rollback restores.
    """
    from src.domain.graph import KnowledgeGraph
    from src.domain.release import ReleaseManifest
    from src.infrastructure.canonical_store import (
        publish_pointer,
        write_manifest,
    )
    from src.infrastructure.graph_artifact import publish_release_graph
    from src.services.admin_rollback import release_tag

    store_root = Path(settings.CANONICAL_STORE_ROOT)
    releases_dir = store_root / "releases"
    releases_dir.mkdir(parents=True, exist_ok=True)
    for rid, parent, table in (
        (RELEASE_A, None, TABLE_A),
        (RELEASE_B, RELEASE_A, TABLE_B),
    ):
        manifest = ReleaseManifest(
            release_id=rid,
            parent_release_id=parent,
            lancedb_table=table,
            lancedb_tag=release_tag(rid),
            chunk_count=chunk_count,
        )
        write_manifest(releases_dir / rid, manifest)

    # Both releases pin a graph. The runtime is built with a
    # GraphRAGRetriever whose ``_graph`` is a non-None (EMPTY)
    # ``KnowledgeGraph``, so ``ReleaseGraphRollback.validate`` refuses a target
    # that pins no graph (TargetGraphNotPinnedError -> 422). An empty graph is
    # honest for the drill: the production graph's nodes are entity ids
    # (``NGANH:*``), so its expander cannot inject this corpus's chunk ids
    # (design §5) — the mixing vector is the booster's second read, not the
    # graph.
    graph_src = workdir / "chaos_t29_graph.graphml"
    KnowledgeGraph.empty().save(graph_src)
    for rid in (RELEASE_A, RELEASE_B):
        publish_release_graph(releases_dir / rid, graph_src)
    publish_pointer(releases_dir / "active_release.json", RELEASE_B)


# ---------------------------------------------------------------------------
# The drill
# ---------------------------------------------------------------------------


async def _run_drill(
    scenario: Dict[str, Any],
    workdir: Path,
) -> DrillResult:
    from config.settings import RAGSettings
    from fastapi.testclient import TestClient
    from rag2025.src.app.lifecycle import configure_canonical_record_store

    from services.aggregation_booster import detect_aggregation_chunks
    from services.graphrag_retriever import GraphRAGRetriever, UnifiedRAGPipeline
    from services.hybrid_search import HybridSearchService
    from services.query_router import SmartQueryRouter
    from services.rate_limit import FixedWindowLimiter
    from src.domain.graph import KnowledgeGraph
    from src.services.lancedb_retrieval import LanceDBRetriever

    result = DrillResult(scenario=dict(scenario))

    # The booster-miss counter is a PROCESS-GLOBAL singleton
    # (``aggregation_booster.BOOSTER_CHUNK_MISS``), so its absolute value is
    # polluted by any earlier test that drove a real query in this process
    # (e.g. test_publish_hook_e2e). The drill's own property is "how many
    # misses did THIS drill's queries cause", so measure the delta across the
    # run instead of the absolute reading.
    _boost_before = main_boost_counter()

    workers = int(scenario["workers"])
    duration_s = float(scenario["duration_s"])
    rate_rps = float(scenario["rate_rps"])
    publish_at_s = float(scenario["publish_at_s"])
    rollback_at_s = float(scenario["rollback_at_s"])
    query_delay_s = float(scenario["query_delay_s"])

    # 1. Real app, no lifespan (design §7).
    main = _load_main()
    _neutralize_lifespan(main)
    settings = RAGSettings()
    configure_canonical_record_store(main.app, settings=settings)

    # 2. Learn the booster's ids from the REAL rule (never hardcoded: they are
    #    year-templated from CURRENT_ADMISSION_YEAR).
    booster_ids = detect_aggregation_chunks(ENUM_QUERY)
    if not booster_ids:
        raise SystemExit("booster rule did not fire for the enumeration query")
    result.booster_ids = list(booster_ids)
    extra_ids = [
        f"chaos_t29_{CURRENT_ADMISSION_YEAR}_extra_{i:02d}" for i in range(8)
    ]
    result.seed_ids = list(booster_ids)
    result.extra_ids = list(extra_ids)

    embedding = FakeEmbedding()
    rows_a = build_release_rows(booster_ids, extra_ids, RELEASE_A, embedding)
    rows_b = build_release_rows(booster_ids, extra_ids, RELEASE_B, embedding)
    _seed_lancedb(workdir, rows_a, rows_b)
    _seed_store(workdir, main, settings, chunk_count=len(rows_a))

    # 3. Real serving retriever over table A (same class the reload route builds).
    retriever = await main.build_retriever_for_table(TABLE_A)
    # Tagging happens inside _seed_lancedb, on the handle that wrote each
    # version; tag_release refuses a tag that already exists.
    # 4. Real router (zero-LLM enumeration fast path) + real booster + real
    #    GraphRAG fusion over an EMPTY graph (the production graph's nodes are
    #    entity ids, not chunk ids, so its expander cannot reach this corpus).
    stub_llm = StubLLM()
    router = SmartQueryRouter(llm=stub_llm)
    # GraphRAGRetriever's real signature is (ner_service, graph, ...) — there is
    # no `llm` parameter (graphrag_retriever.py:170-179). An EMPTY graph keeps
    # `graph_stats` readable at payload build time (main.py:1939) while the stub
    # NER keeps PPR unseeded, so fusion adds nothing to the served set.
    graphrag = GraphRAGRetriever(
        ner_service=StubNER(),
        graph=KnowledgeGraph.empty(),
        chunk_fetcher=lambda cid: retriever.fetch_by_id(cid),
    )
    pipeline = UnifiedRAGPipeline(
        router=router, graphrag=graphrag, llm=StubLLM()
    )

    main.embedding_service = embedding
    main.reranker_service = None  # absent reranker => adaptive rerank gate skipped
    main.unified_pipeline = pipeline
    main.hybrid_search_service = None
    main.query_cache = None  # design §9
    main.guardrail_service = None
    main.lancedb_retriever_service = retriever
    main.llm_generator_service = StubGenerator()
    main._query_rate_limiter = FixedWindowLimiter(limit=10_000_000)
    main._enforce_rate_limit = _no_rate_limit  # design §8 (braces)
    # RBAC is resolved from the ENVIRONMENT, not from app.state: `current_identity`
    # calls `resolve_identity` -> `load_identities()`, which reads
    # ADMIN_IDENTITIES/ADMIN_API_TOKEN (src/config/admin_identity.py:127-148).
    # Writing app.state.admin_identities would authenticate nobody.
    #
    # The registry must hold TWO identities, because the two injections need
    # different grants and no single role has both:
    #   reload-table -> OPS_RELOAD_TABLE
    #   rollback     -> RECORDS_ROLLBACK, which Role.OPS does NOT hold
    #                   (ROLE_MATRIX, src/domain/admin_identity.py:138-150).
    # A single ops token would make the rollback injection 403 and the drill
    # would report "no mixed answers" without ever rolling anything back.
    #
    # OPS_TOKEN is deliberately NOT listed as a human actor here. The ops
    # credential is injected by the registry itself as the machine identity
    # (OPS_ACTOR_ID, Role.OPS), and registering the same string as an actor too
    # trips the separation-of-duties guard in load_identities(): "the shared ops
    # credential must not double as a human actor". Role.ADMIN already holds
    # both OPS_RELOAD_TABLE and RECORDS_ROLLBACK, so one admin actor serves
    # both injections and the guard stays armed.
    os.environ["ADMIN_IDENTITIES"] = json.dumps(
        [
            {"token": ADMIN_TOKEN, "actor_id": "chaos.admin@husc", "role": "admin"},
        ]
    )
    os.environ["ADMIN_API_TOKEN"] = OPS_TOKEN
    main._publish_retrieval_aliases(main.app.state)
    _wire_app_state(main, embedding, None)
    pipeline._router = router
    pipeline._graphrag = graphrag
    pipeline.hybrid_search_service = None

    ops_headers = {"X-Admin-Token": OPS_TOKEN}
    admin_headers = {"X-Admin-Token": ADMIN_TOKEN}

    # 5. Load. One event loop, N async workers (TestClient is loop-safe here).
    t0 = time.monotonic()
    inflight: List[asyncio.Task] = []
    interval = 1.0 / rate_rps if rate_rps > 0 else 0.0
    n_requests = max(1, int(duration_s * rate_rps)) if rate_rps > 0 else workers
    sem = asyncio.Semaphore(workers)
    client_box: Dict[str, Any] = {}

    def _make_client() -> Any:
        return TestClient(main.app)

    async def fire(index: int) -> None:
        async with sem:
            # `rate_rps` is a FREQUENCY, not just a request count: without this
            # stagger every request would be launched at t=0 and the drill would
            # never actually overlap the swap window it exists to probe.
            if interval > 0:
                await asyncio.sleep(index * interval)
            if query_delay_s > 0:
                await asyncio.sleep(query_delay_s)
            client = client_box.setdefault("c", _make_client())
            loop = asyncio.get_running_loop()
            # Sample the tracker BEFORE the request goes out: `_inflight` is a
            # key no route emits (grep "_inflight" src/ -> nothing), so reading
            # it from the response body left this count permanently at the -1
            # sentinel. The tracker is the only real source, and `in_flight` is
            # a plain counter read (request_tracker.py:140) - safe to touch from
            # the drill's thread because the increment happens inside the app.
            inflight_ok = int(getattr(main.app.state.request_tracker, "in_flight", -1))
            offset_s = time.monotonic() - t0
            status, payload, latency_ms = await loop.run_in_executor(
                None, _one_query, client, index, ENUM_QUERY
            )
            releases = detect_mixed_release(str(payload.get("answer", ""))) if status == 200 else []
            record = {
                "n": index,
                "offset_s": round(offset_s, 4),
                "status": status,
                "latency_ms": round(latency_ms, 3),
                "in_flight_at_send": inflight_ok,
                "releases": releases,
                "route": payload.get("route") if isinstance(payload, dict) else None,
                "query": payload.get("query") if isinstance(payload, dict) else None,
                "n_chunks": len(payload.get("sources") or []) if isinstance(payload, dict) else 0,
            }
            result.requests.append(record)
            if status != 200:
                body = json.dumps(payload, ensure_ascii=False)[:400]
                record["body"] = body

    async def scheduled(
        method: str,
        path: str,
        label: str,
        at_s: float,
        body: Dict[str, Any],
        headers: Dict[str, str],
    ) -> None:
        await asyncio.sleep(at_s)
        client = client_box.setdefault("c", _make_client())
        loop = asyncio.get_running_loop()
        t_send = time.monotonic() - t0
        status, payload = await loop.run_in_executor(
            None, _one_admin, client, method, path, body, headers
        )
        result.injections.append(
            {
                "label": label,
                "method": method,
                "path": path,
                "body": body,
                "at_s": at_s,
                "send_offset_s": round(t_send, 4),
                "status": status,
                "response": payload,
                "duration_ms": round((time.monotonic() - t0 - t_send) * 1000.0, 3),
            }
        )

    tasks = [asyncio.create_task(fire(i)) for i in range(n_requests)]
    tasks.append(
        asyncio.create_task(
            scheduled(
                "POST",
                "/admin/reload-table",
                "publish",
                publish_at_s,
                {
                    "table_name": TABLE_B,
                    "drain_timeout_s": 10.0,
                    # The drill keeps firing until duration_s, and ``gate()``
                    # closes admission for the whole drain (request_tracker.py:
                    # track() waits on _admit_event BEFORE incrementing). Under
                    # sustained load the counter therefore cannot reach 0, so a
                    # non-forced reload can only ever answer 409 — and a refused
                    # swap exercises no seam at all, which is the vacuity the
                    # ticket's "0 mixed" criterion has to be protected from.
                    # Forcing is the documented path for swapping under load
                    # (admin.py:132-136): admission is STILL closed for the swap
                    # itself, so nothing new is admitted into the half-swapped
                    # window, and any read that does span two releases is what
                    # the detector exists to catch.
                    "force_while_in_flight": True,
                },
                ops_headers,
            )
        )
    )
    tasks.append(
        asyncio.create_task(
            scheduled(
                "POST",
                "/admin/rollback",
                "rollback",
                rollback_at_s,
                {"target_release_id": RELEASE_A, "reason": "ticket-29 chaos drill"},
                admin_headers,
            )
        )
    )

    # One ENTERED client => one portal => one event loop shared by every request
    # and both admin injections, which is what makes the drain real. The lifespan
    # the entry would normally run is a no-op (`_neutralize_lifespan`), so the
    # stubs above survive it.
    with _count_fetch_by_id() as fetch_calls:
        with _make_client() as probe_client:
            client_box.setdefault("c", probe_client)
            await asyncio.gather(*tasks)

    # 6. Aggregate.
    window_lo = min(publish_at_s, rollback_at_s)
    window_hi = max(publish_at_s, rollback_at_s)
    # Two different questions, two different sets. "Sent during the window" is
    # structurally empty: `gate()` closes admission for the whole drain
    # (request_tracker.py:104), so nothing is admitted while a swap is in
    # progress. The set that can actually observe a half-swapped runtime is the
    # one that was ALREADY in flight when the window opened and finished after
    # it - so classify by the request's own interval, not its send instant.
    # Reporting only the first set makes the latency criterion look permanently
    # unmet (n=0, p50/p95 null) when the real answer is in the second.
    def _overlaps(rec: Dict[str, Any]) -> bool:
        start = float(rec["offset_s"])
        end = start + float(rec["latency_ms"]) / 1000.0
        return start <= window_hi and end >= window_lo

    inside = [r for r in result.requests if _overlaps(r)]
    outside = [r for r in result.requests if not _overlaps(r)]
    sent_in_window = [
        r for r in result.requests if window_lo <= r["offset_s"] <= window_hi
    ]
    result.latency = {
        "swap_window": {"window_s": [window_lo, window_hi], **_latency_stats(inside)},
        "sent_in_window": _latency_stats(sent_in_window),
        "outside_window": _latency_stats(outside),
        "all": _latency_stats(result.requests),
    }
    result.errors_by_status = {
        str(code): n
        for code, n in sorted(Counter(r["status"] for r in result.requests).items())
        if code != 200
    }
    result.mixed = [r for r in result.requests if len(r["releases"]) > 1]
    result.counts.update(
        {
            "requests": len(result.requests),
            "http_200": sum(1 for r in result.requests if r["status"] == 200),
            "http_non_200": sum(1 for r in result.requests if r["status"] != 200),
            "in_window": len(inside),
            "outside_window": len(outside),
            "mixed_release_answers": len(result.mixed),
            "router_llm_calls": stub_llm.calls,
            "generator_calls": main.llm_generator_service.calls,
            "booster_chunk_miss": main_boost_counter() - _boost_before,
            "fetch_by_id_calls": fetch_calls["n"],
            "max_in_flight_observed": max(
                (r["in_flight_at_send"] for r in result.requests), default=-1
            ),
        }
    )
    return result


def main_boost_counter() -> int:
    from services.aggregation_booster import BOOSTER_CHUNK_MISS

    return int(BOOSTER_CHUNK_MISS.get())


def _one_query(client: Any, index: int, base_query: str) -> Tuple[int, Dict[str, Any], float]:
    nonce = f" [q{index:05d}]"  # design §9: identical text would measure cache hits
    payload = {
        "query": base_query + nonce,
        "top_k": 5,
        "force_route": "padded_rag",
        "history": [],
    }
    t = time.perf_counter()
    try:
        resp = client.post("/v2/query", json=payload)
    except Exception as exc:  # transport-level failure is a result, not a crash
        return 0, {"detail": f"{type(exc).__name__}: {exc}"}, (time.perf_counter() - t) * 1000.0
    latency_ms = (time.perf_counter() - t) * 1000.0
    try:
        body = resp.json()
    except Exception:
        body = {"detail": resp.text[:400]}
    if not isinstance(body, dict):
        body = {"detail": str(body)[:400]}
    return resp.status_code, body, latency_ms


def _one_admin(client: Any, method: str, path: str, body: Dict[str, Any], headers: Dict[str, str]) -> Tuple[int, Any]:
    try:
        resp = client.request(method, path, json=body, headers=headers)
    except Exception as exc:
        return 0, {"detail": f"{type(exc).__name__}: {exc}"}
    try:
        return resp.status_code, resp.json()
    except Exception:
        return resp.status_code, {"detail": resp.text[:400]}


def verdict_ok(result: DrillResult) -> bool:
    """Whether the drill passed, as a boolean of the SAME conditions
    :func:`verdict_line` renders. Kept beside it so the exit code and the
    printed verdict cannot drift apart."""
    if not all(i["status"] == 200 for i in result.injections):
        return False
    n_mixed = result.counts.get("mixed_release_answers", 0)
    if result.expect_mixed:
        # Mutation proof: it MUST observe the mixed answer it forced, otherwise
        # the detector is unproven and every "0 mixed" claim is unfalsifiable.
        if not n_mixed or not result.counts.get("poisoned_reads", 0):
            return False
    elif n_mixed:
        return False
    return result.counts.get("http_200", 0) > 0


def verdict_line(result: DrillResult) -> str:
    n_200 = result.counts.get("http_200", 0)
    n_bad = result.counts.get("http_non_200", 0)
    n_mixed = result.counts.get("mixed_release_answers", 0)
    inj_ok = all(i["status"] == 200 for i in result.injections)
    inj_desc = ", ".join(f"{i['label']}={i['status']}" for i in result.injections)
    if not inj_ok:
        return f"FAIL — injection(s) did not complete: {inj_desc}"
    if result.expect_mixed:
        if not result.counts.get("poisoned_reads", 0):
            return (
                "FAIL — mutation proof never poisoned a read: "
                "`fetch_by_id` was not reached, so the detector is untested"
            )
        if not n_mixed:
            return (
                "FAIL — detector did not fire on a forced half-swap "
                "(two release ids in one response were expected)"
            )
        return (
            f"PASS — detector fired: {n_mixed} mixed-release answer(s) from "
            f"{result.counts.get('poisoned_reads')} poisoned read(s); the gate "
            f"is capable of failing, so a 0-mixed drill result is meaningful"
        )
    if n_mixed:
        return (
            f"FAIL — {n_mixed} mixed-release answer(s) observed under load "
            f"({inj_desc}); see `mixed_release_answers`"
        )
    if n_200 == 0:
        return f"FAIL — no successful query under load (non-200: {result.errors_by_status})"
    extra = f", {n_bad} non-200" if n_bad else ", zero non-200"
    return (
        f"PASS — {n_mixed} mixed-release answers across {n_200} successful "
        f"concurrent /v2/query calls while a real publish and a real rollback "
        f"swapped the serving table ({inj_desc}){extra}"
    )


# ---------------------------------------------------------------------------
# Mutation proof (design §Remaining-work 5)
# ---------------------------------------------------------------------------


async def _run_mutation_proof(workdir: Path) -> DrillResult:
    """Force two release ids into ONE response and show the detector fires.

    Same app as the drill, but the serving retriever's ``fetch_by_id`` is
    patched so the booster's second read returns the *other* release's row for
    one chunk — exactly the half-swap the gate is supposed to make impossible.
    Without this, a zero mixed-release count proves nothing about the detector.
    """
    from fastapi.testclient import TestClient

    scenario = {
        "workers": 1,
        "duration_s": 0.0,
        "rate_rps": 0.0,
        "publish_at_s": 1e9,
        "rollback_at_s": 1e9,
        "query_delay_s": 0.0,
        "seed": 29,
    }
    result = await _build_quiet_app(scenario, workdir)
    result.expect_mixed = True
    main = _MAIN_BOX["main"]
    retriever = main.lancedb_retriever_service
    original = retriever.fetch_by_id
    victim = result.seed_ids[0]

    hits = {"calls": 0, "poisoned": 0}

    def poisoned(chunk_id: str):
        """Return the OTHER release's row for the victim chunk.

        ``RetrievedDocument`` is a FROZEN dataclass (``lancedb_retrieval.py:75``),
        so in-place assignment (``doc.text = ...``) raises ``FrozenInstanceError``.
        The booster calls this inside its own ``try/except Exception``
        (``aggregation_booster.py:131-160``), so that error is swallowed, the row
        is never poisoned, and the proof passes having injected nothing. Rebuild
        through ``replace`` — the same way the booster itself does.
        """
        hits["calls"] += 1
        doc = original(chunk_id)
        if doc is None or chunk_id != victim:
            return doc
        current = (doc.metadata or {}).get("release_id")
        other = RELEASE_B if current == RELEASE_A else RELEASE_A
        hits["poisoned"] += 1
        return replace(
            doc,
            text=doc.text.replace(str(current), other),
            metadata={**(doc.metadata or {}), "release_id": other},
        )

    retriever.fetch_by_id = poisoned
    try:
        with TestClient(main.app) as client:
            status, payload, latency_ms = _one_query(client, 1, ENUM_QUERY)
    finally:
        retriever.fetch_by_id = original

    releases = detect_mixed_release(str(payload.get("answer", ""))) if status == 200 else []
    record = {
        "n": 1,
        "offset_s": 0.0,
        "status": status,
        "latency_ms": round(latency_ms, 3),
        "in_flight_at_send": -1,
        "releases": releases,
        "route": payload.get("route"),
        "query": payload.get("query"),
        "n_chunks": len(payload.get("sources") or []),
        "mutated_chunk_id": victim,
    }
    result.requests = [record]
    result.mixed = [record] if len(releases) > 1 else []
    result.errors_by_status = {} if status == 200 else {str(status): 1}
    result.counts.update(
        {
            "requests": 1,
            "http_200": 1 if status == 200 else 0,
            "http_non_200": 0 if status == 200 else 1,
            "in_window": 0,
            "outside_window": 1,
            "mixed_release_answers": len(result.mixed),
            "fetch_by_id_calls": hits["calls"],
            "poisoned_reads": hits["poisoned"],
        }
    )
    result.latency = {
        "swap_window": {"window_s": [], **_latency_stats([])},
        "outside_window": _latency_stats(result.requests),
        "all": _latency_stats(result.requests),
    }
    result.injections = [
        {
            "label": "mutation-proof",
            "method": "patch",
            "path": "LanceDBRetriever.fetch_by_id",
            "body": {"chunk_id": victim, "forced_release": "opposite"},
            "at_s": 0.0,
            "send_offset_s": 0.0,
            "status": 200,
            "response": {"observed_release_ids": releases},
            "duration_ms": 0.0,
        }
    ]
    return result


_MAIN_BOX: Dict[str, Any] = {}


async def _build_quiet_app(scenario: Dict[str, Any], workdir: Path) -> DrillResult:
    """Construct the drill app without firing load (mutation proof reuses it)."""
    from config.settings import RAGSettings
    from rag2025.src.app.lifecycle import configure_canonical_record_store

    from services.aggregation_booster import detect_aggregation_chunks
    from services.graphrag_retriever import GraphRAGRetriever, UnifiedRAGPipeline
    from services.query_router import SmartQueryRouter
    from services.rate_limit import FixedWindowLimiter
    from src.domain.graph import KnowledgeGraph

    result = DrillResult(scenario=dict(scenario))
    main = _load_main()
    _neutralize_lifespan(main)
    _MAIN_BOX["main"] = main
    settings = RAGSettings()
    configure_canonical_record_store(main.app, settings=settings)

    booster_ids = detect_aggregation_chunks(ENUM_QUERY)
    extra_ids = [f"chaos_t29_{CURRENT_ADMISSION_YEAR}_extra_{i:02d}" for i in range(8)]
    result.seed_ids = list(booster_ids)
    result.extra_ids = list(extra_ids)
    result.booster_ids = list(booster_ids)

    embedding = FakeEmbedding()
    _seed_lancedb(
        workdir,
        build_release_rows(booster_ids, extra_ids, RELEASE_A, embedding),
        build_release_rows(booster_ids, extra_ids, RELEASE_B, embedding),
    )
    _seed_store(workdir, main, settings)
    retriever = await main.build_retriever_for_table(TABLE_A)

    stub_llm = StubLLM()
    router = SmartQueryRouter(llm=stub_llm)
    graphrag = GraphRAGRetriever(
        ner_service=StubNER(),
        graph=KnowledgeGraph.empty(),
        chunk_fetcher=lambda cid: retriever.fetch_by_id(cid),
    )
    pipeline = UnifiedRAGPipeline(router=router, graphrag=graphrag, llm=StubLLM())
    main.embedding_service = embedding
    main.reranker_service = None
    main.unified_pipeline = pipeline
    main.hybrid_search_service = None
    main.query_cache = None
    main.guardrail_service = None
    main.lancedb_retriever_service = retriever
    main.llm_generator_service = StubGenerator()
    main._query_rate_limiter = FixedWindowLimiter(limit=10_000_000)
    main._enforce_rate_limit = _no_rate_limit
    os.environ["ADMIN_IDENTITIES"] = json.dumps(
        [
            {"token": ADMIN_TOKEN, "actor_id": "chaos.admin@husc", "role": "admin"},
        ]
    )
    os.environ["ADMIN_API_TOKEN"] = OPS_TOKEN
    main._publish_retrieval_aliases(main.app.state)
    _wire_app_state(main, embedding, None)
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_artifacts(out_dir: Path, result: DrillResult, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ticket": "29",
        "artifact": tag,
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "caveat": result.caveat,
        "scenario": result.scenario,
        "environment": {
            "python": sys.executable,
            "table_before_publish": TABLE_A,
            "table_published": TABLE_B,
            "release_a": RELEASE_A,
            "release_b": RELEASE_B,
            "vector_dim": result.vector_dim,
            "admission_year": CURRENT_ADMISSION_YEAR,
            "enumeration_query": ENUM_QUERY,
            "booster_ids_from_real_rule": result.booster_ids,
            "seeded_extra_ids": result.extra_ids,
            "hybrid_retrieval": "off (design §10)",
            "query_cache": "disabled + per-request nonce (design §9)",
            "rate_limiter": "bypassed (design §8)",
            "router_llm_calls": result.counts.get("router_llm_calls"),
            "generator": "STUBBED (provider down)",
        },
        "injections": result.injections,
        "counts": result.counts,
        "errors_by_status": result.errors_by_status,
        "latency": result.latency,
        "mixed_release_answers": result.mixed,
        "requests": result.requests,
        "verdict": verdict_line(result),
    }
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _print_summary(result: DrillResult, path: Path) -> None:
    c = result.counts
    print(f"=== {path.name} ===")
    print(f"scenario: {json.dumps(result.scenario, sort_keys=True)}")
    print(f"seeded booster ids (real rule): {result.booster_ids}")
    print(f"requests={c.get('requests')} 200={c.get('http_200')} non200={c.get('http_non_200')} "
          f"errors_by_status={result.errors_by_status}")
    print(f"in-flight max observed={c.get('max_in_flight_observed')}  "
          f"router LLM calls={c.get('router_llm_calls')}  generator calls={c.get('generator_calls')}  "
          f"booster chunk misses={c.get('booster_chunk_miss')}")
    for inj in result.injections:
        print(f"injection {inj['label']}: HTTP {inj['status']} at t={inj['at_s']}s "
              f"({json.dumps(inj['response'], ensure_ascii=False)[:300]})")
    lw = result.latency["swap_window"]
    lo = result.latency["outside_window"]
    print(f"latency inside swap window {lw['window_s']}: n={lw['n_200']} "
          f"p50={lw['p50_ms']} p95={lw['p95_ms']} max={lw['max_ms']}")
    print(f"latency outside window:            n={lo['n_200']} p50={lo['p50_ms']} "
          f"p95={lo['p95_ms']} max={lo['max_ms']}")
    print(f"mixed-release answers: {c.get('mixed_release_answers')}")
    for m in result.mixed[:5]:
        print(f"  n={m['n']} releases={m['releases']} query={m['query']!r}")
    print(f"VERDICT: {verdict_line(result)}")
    print(f"CAVEAT: {result.caveat}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", type=str, default=None, help="path to scenario JSON")
    parser.add_argument("--out", type=str, default=str(RAG2025 / "logs" / "chaos-drill"))
    parser.add_argument("--tag", type=str, default="drill")
    parser.add_argument("--mutation-proof", action="store_true")
    args = parser.parse_args(argv)


    scenario = dict(DEFAULT_SCENARIO)
    if args.scenario:
        loaded = json.loads(Path(args.scenario).read_text(encoding="utf-8"))
        unknown = set(loaded) - set(DEFAULT_SCENARIO)
        if unknown:
            raise SystemExit(f"unknown scenario keys: {sorted(unknown)}")
        scenario.update(loaded)

    out_dir = Path(args.out)
    workdir = out_dir / f"work-{args.tag}"
    if workdir.exists():
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    _configure_env(workdir)

    if args.mutation_proof:
        result = asyncio.run(_run_mutation_proof(workdir))
        path = _write_artifacts(out_dir, result, f"{args.tag}-mutation-proof")
        _print_summary(result, path)
        return 0 if verdict_ok(result) else 1

    result = asyncio.run(_run_drill(scenario, workdir))
    path = _write_artifacts(out_dir, result, args.tag)
    _print_summary(result, path)
    return 0 if verdict_ok(result) else 1


if __name__ == "__main__":
    raise SystemExit(main())
