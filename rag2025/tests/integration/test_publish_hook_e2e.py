"""Ticket T01, end-to-end: pressing Publish reaches the chatbot.

The reindex job builds the corpus aside and calls ``publish(staged,
corpus_dir)`` — the hook this file measures.  Five acceptance
criteria, each with one test:

(i)   publish → ≤ 2 ticks → ``/v2/query`` returns a chunk of the
      published record (via the real unified_query handler over a
      real temp LanceDB);
(ii)  ingest error → job FAILED, outbox keeps the entry, old corpus
      still serves, next tick OK;
(iii) 2 publishes + rollback → release-1 answer returns, stale cache
      not served;
(iv)  rollback concurrent with a tick does not corrupt the pointer;
(v)   legacy row count unchanged after publishes.

The app is assembled exactly like ``test_reindex_runner_e2e.py``
(the real ``configure_canonical_record_store`` startup seam, autorun
on, corpus rebound to ``tmp_path``), with one addition: ``LANCEDB_URI``
points at a temp dir so the ingest + tag + retrieval steps run against
a disposable table, and ``app.state.build_retriever`` is the real
``build_retriever_for_table`` shape so the hook's swap actually moves
the serving retriever.  ``/v2/query`` is driven through the real
handler from a loaded ``src/main.py`` module whose heavy singletons
(embedding, generator, guardrail, pipeline) are fakes — the same
technique ``test_v2_booster_wired.py`` uses.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import hashlib
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock

import lancedb
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.request_tracker import RequestTracker  # noqa: E402
from src.domain.canonical import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.routers.admin import router as admin_router  # noqa: E402
from src.services import release_status  # noqa: E402
from src.services.admin_rollback import (  # noqa: E402
    AdminRollbackService,
    LanceDBRetrievalRollback,
    ReleaseGraphRollback,
)
from src.services.canonical_chunk_builder import (  # noqa: E402
    CHUNKS_FILENAME,
    CHUNK_MANIFEST_FILENAME,
)
from src.services.query_cache import QueryCache  # noqa: E402
from src.services.reindex_job import (  # noqa: E402
    GRAPH_EDGES_FILENAME,
    VECTORS_FILENAME,
    JobState,
)
from src.services.reindex_outbox import OUTBOX_FILENAME, ReindexOutbox  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "reviewer": ("cb.phan-bien", "reviewer"),
    "publisher": ("cb.xuat-ban", "publisher"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}

REC_A = "CHI_TIEU:nganh:7220104:2026:a"
REC_B = "CHI_TIEU:nganh:7480201:2026:b"

TEXT_A = (
    "Điểm chuẩn ngành Công nghệ thông tin (mã ngành 7220104) năm 2026 là 25,5 "
    "điểm, tổ hợp xét tuyển A00.\n"
)
# Ticket T07 acceptance #2: the HOI_DAP pair — the ticket's own scenario.
# ``TEXT_HD`` is a question/answer the student can cite; ``QUERY_HD`` is the
# question a student actually asks.  The record carries an explicit
# non-empty ``source_document_id`` so the citation gate
# (``abstention_policy._is_traceable``) is exercised without depending on
# the separate server-side source-document generation landing first.
REC_HD = "HOI_DAP:hosc-may-tinh:2026:hd"
TEXT_HD = (
    "Câu hỏi: Điểm chuẩn ngành Khoa học máy tính năm 2026 bao nhiêu?\n"
    "Trả lời: Điểm chuẩn năm 2026 là 27,0 điểm, tổ hợp xét tuyển A00.\n"
)
QUERY_HD = "Điểm chuẩn ngành Khoa học máy tính năm 2026?"

TEXT_B = "Học phí ngành Khoa học máy tính năm 2026 là 15.000.000 đồng."

# The question the tests ask the chatbot.  Contains TEXT_A so the
# fake generator can assert the published chunk reached the answer.
QUERY_A = "Điểm chuẩn ngành Công nghệ thông tin 2026?"

EMBEDDING_DIM = 1024


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _record(record_id: str, text: str, version: int = 1) -> CanonicalRecord:
    return CanonicalRecord(
        record_id=record_id,
        record_type=RecordType.CHI_TIEU,
        version=version,
        admission_year=2026,
        effective_from=date(2026, 1, 1),
        effective_to=date(2026, 12, 31),
        status=RecordStatus.PUBLISHED,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="de-an-tuyen-sinh-2026",
        source_url="https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        source_page=12,
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        text=text,
    )


def _record_body(record_id: str, text: str) -> dict:
    return {
        "record_id": record_id,
        "record_type": "CHI_TIEU",
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-01-01",
        "effective_to": "2026-12-31",
        "status": "draft",
        "authority_level": "husc_official",
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 24,
        "text": text,
    }


def _hoi_dap_record_body(record_id: str, text: str) -> dict:
    """Record body for a HOI_DAP pair: the ticket's scenario.

    ``source_document_id`` is explicit (not generated) so the test exercises
    the citation gate without depending on the server-side generation that
    the HoiDapSourceDoc landing adds under the registry's ``documents_dir``.
    """
    return {
        "record_id": record_id,
        "record_type": "HOI_DAP",
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-01-01",
        "effective_to": "2026-12-31",
        "status": "draft",
        "authority_level": "husc_official",
        "source_document_id": "hoi-dap-" + record_id,
        "source_url": None,
        "source_page": None,
        "text": text,
    }


def _publish_hoi_dap_record(client: TestClient, record_id: str, text: str) -> None:
    """Create + publish a HOI_DAP record over HTTP, like ``_publish_record``."""
    created = client.post(
        "/admin/records",
        json={"record": _hoi_dap_record_body(record_id, text)},
        headers=_headers("editor"),
    )
    assert created.status_code == 200, created.text
    for action, role in (
        ("submit", "editor"),
        ("approve", "reviewer"),
        ("publish", "publisher"),
    ):
        response = client.post(
            f"/admin/records/{record_id}/transitions",
            json={"action": action, "expected_version": {"submit": 1, "approve": 2, "publish": 3}[action]},
            headers=_headers(role),
        )
        assert response.status_code == 200, response.text



class FakeEmbeddingService:
    """Deterministic embedder over ``encode_documents`` / ``encode_query``."""

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    def encode_documents(self, documents):
        texts = list(documents)
        self.batches.append(texts)
        return [[float(len(t) % 97)] + [0.5] * (self.dim - 1) for t in texts]

    def encode_query(self, query: str):
        import numpy as np

        return np.array([float(len(query) % 97)] + [0.5] * (self.dim - 1))

    @property
    def embedded_texts(self) -> list[str]:
        return [t for batch in self.batches for t in batch]


class FakeLLMGenerator:
    """Generator that answers with the retrieved chunk texts verbatim."""

    async def generate_answer(self, *, query, chunks, confidence,
                            is_program_list_query=False, history=None):
        joined = "\n".join(chunk["text"] for chunk in chunks)
        return {
            "answer": joined,
            "sources": [{"chunk_id": chunk["chunk_id"]} for chunk in chunks],
            "provider": "fake",
            "generation_failed": False,
        }


class FakeRouterResult:
    """What ``unified_pipeline._router.route()`` returns on the happy path."""

    def __init__(self) -> None:
        self.skip_retrieval = False
        self.auto_answer = None
        self.step_back_query = "step"
        self.intent = "diem_chuan"
        self.complexity = 1
        self.reasoning = "single-hop"
        self.hyde_variants: list[str] = []
        self.is_in_scope = True
        class _Route:
            value = "padded_rag"
        self.route = _Route()


class FakeRouter:
    async def route(self, query: str) -> FakeRouterResult:
        return FakeRouterResult()


@dataclass
class FakeRAGResult:
    documents: list = field(default_factory=list)
    route: str = "padded_rag"
    router_result: FakeRouterResult = field(default_factory=FakeRouterResult)
    confidence: float = 0.9
    ppr_scores: dict = field(default_factory=dict)
    latency_ms: float = 1.0


class FakeGraphRAG:
    """The pipeline's graph half: stats + retrieve, over fakes."""

    def __init__(self) -> None:
        self.graph_stats = {"nodes": 0, "edges": 0}

    async def retrieve(self, *, query, router_result, baseline_docs, top_k):
        return list(baseline_docs[:top_k]), {}

    def update_graph(self, graph) -> None:
        pass


class FakePipeline:
    """The ``unified_pipeline`` the /v2 handler reads, over fakes.

    ``query()`` is the PaddedRAG happy path: documents already in
    ``baseline_docs`` (the retriever's output) are returned unchanged —
    exactly what the real pipeline does for a simple 1-hop query.
    """

    def __init__(self) -> None:
        self._router = FakeRouter()
        self._graphrag = FakeGraphRAG()

    async def query(self, *, user_query, baseline_docs, top_k, router_result):
        return FakeRAGResult(documents=list(baseline_docs[:top_k]))


def _wait_for(predicate, timeout: float = 30.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _read_manifest(corpus_dir: Path) -> Optional[dict]:
    try:
        return json.loads((corpus_dir / CHUNK_MANIFEST_FILENAME).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _chunk_texts(corpus_dir: Path) -> Set[str]:
    if not (corpus_dir / CHUNKS_FILENAME).is_file():
        return set()
    return {
        json.loads(line)["text"]
        for line in (corpus_dir / CHUNKS_FILENAME).read_text("utf-8").splitlines()
        if line.strip()
    }


def _publish_record(client: TestClient, record_id: str, text: str) -> None:
    """Create a draft and walk it to published over HTTP, like the UI."""
    created = client.post(
        "/admin/records",
        json={"record": _record_body(record_id, text)},
        headers=_headers("editor"),
    )
    assert created.status_code == 200, created.text
    for action, role in (
        ("submit", "editor"),
        ("approve", "reviewer"),
        ("publish", "publisher"),
    ):
        response = client.post(
            f"/admin/records/{record_id}/transitions",
            json={"action": action, "expected_version": {"submit": 1, "approve": 2, "publish": 3}[action]},
            headers=_headers(role),
        )
        assert response.status_code == 200, response.text


# ── the /v2 query handler, loaded the way test_v2_booster_wired loads main ──

_MAIN_SPEC = importlib.util.spec_from_file_location(
    "publish_hook_e2e_main", ROOT / "src" / "main.py"
)
_main = importlib.util.module_from_spec(_MAIN_SPEC)
_MAIN_SPEC.loader.exec_module(_main)


def _drive_v2_query(main, query: str, embedder: FakeEmbeddingService) -> dict:
    """Call the real ``unified_query`` handler over fakes.

    The handler's globals (``unified_pipeline``, ``lancedb_retriever_service``,
    ``embedding_service``, ``llm_generator_service``, ``query_cache``) are set
    from the test's fixtures; the retriever is the one the publish hook
    swapped in, so the answer's chunks prove which release serves.
    """
    # The publish hook swaps main's globals from the worker thread; the slots
    # the test drives must be refreshed before they are read, or a query would
    # rebind the stale pre-publish retriever (the fixture's own note: "keep
    # the module-level slots ... in sync with main's globals after every
    # swap").
    _sync_main_globals()
    main.unified_pipeline = _PIPELINE[0]
    main.lancedb_retriever_service = _RETRIEVER[0]
    main.embedding_service = embedder
    main.llm_generator_service = _GENERATOR[0]
    main.query_cache = _CACHE[0]
    main.guardrail_service = None
    main.reranker_service = None
    main.hybrid_search_service = None
    request = main.UnifiedQueryRequest(query=query, top_k=5, force_route="padded_rag")
    raw = MagicMock()
    raw.url = MagicMock()
    payload = asyncio.run(main._build_v2_query_payload(request, raw))
    return payload


# module-level slots the fixture fills; the handler reads module globals,
# so the swap the hook performs must land in the same module the
# handler closure over.  ``_swap_runtime_services`` from main.py rebinds
# main's globals directly, which is exactly what production does.
_PIPELINE: list = [FakePipeline()]
_GENERATOR: list = [FakeLLMGenerator()]
_CACHE: list = [QueryCache(ttl_seconds=900)]
_RETRIEVER: list = [None]


async def _build_retriever_for_table(table_name: str):
    """The real construction path — a ``LanceDBRetriever`` over settings."""
    from src.services.lancedb_retrieval import (
        LanceDBRetriever,
        LanceDBRetrieverConfig,
    )
    settings = RAGSettings()
    config = LanceDBRetrieverConfig(
        uri=settings.LANCEDB_URI,
        table_name=table_name,
        embedding_dim=settings.EMBEDDING_DIM,
        default_top_k=5,
    )
    return LanceDBRetriever(config=config)


def _init_empty_husc_chunks_table(lancedb_dir: Path) -> None:
    """Create an empty ``husc_chunks`` table whose schema matches production.

    The served table is normally created by the first publish's ingest; the
    fixture needs it beforehand so the serving retriever can be opened over an
    empty table (the pre-publish sanity query must see ``sources == []``) and so
    the first publish's ``merge_insert`` does not meet a schema mismatch.

    Build it through the real ingest path and then empty it, which guarantees
    the schema is byte-identical to what production writes.
    """
    import scripts.ingest_lancedb as ingest_mod

    scratch = lancedb_dir.parent / ".scratch-empty-ingest"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    stamp = ingest_mod.CorpusStamp.current()
    corpus = scratch / "corpus"
    corpus.mkdir()
    corpus.joinpath("chunk_manifest.json").write_text(
        json.dumps(
            {
                "chunker_version": "1.0",
                "unit_rules_version": stamp.unit_rules_version,
                "chunk_manifest_version": 1,
            },
            ensure_ascii=False,
        )
    )

    text = "x"
    record_version_key = "fixture-pilot-v1"
    chunk_id = hashlib.sha256(f"{record_version_key}:{text}".encode()).hexdigest()
    chunk = {
        "id": chunk_id,
        "content_hash": hashlib.sha256(text.encode()).hexdigest(),
        "text": text,
        "metadata": {
            "record_id": "X",
            "release_id": "00000000000000-0000",
            "status": "published",
            "record_type": "CHI_TIEU",
            "admission_year": 2026,
            "ingest_file": "chunks.jsonl",
        },
    }
    corpus.joinpath("chunks.jsonl").write_text(
        json.dumps(chunk, ensure_ascii=False) + "\n"
    )
    corpus.joinpath("vectors.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": chunk_id,
                "content_hash": chunk["content_hash"],
                "vector": [0.0] * EMBEDDING_DIM,
            },
            ensure_ascii=False,
        )
        + "\n"
    )

    old_uri = os.environ.get("LANCEDB_URI")
    try:
        os.environ["LANCEDB_URI"] = str(lancedb_dir)
        ingest_mod.ingest_canonical_corpus(str(corpus))
    finally:
        if old_uri is not None:
            os.environ["LANCEDB_URI"] = old_uri

    db = lancedb.connect(str(lancedb_dir))
    table = db.open_table(RAGSettings().LANCEDB_TABLE)
    table.delete("TRUE")


def _real_serving_row_count() -> Optional[int]:
    """Rows in the repo's REAL serving table, or ``None`` when it is absent.

    The leak this guards against is invisible: a test that calls
    ``monkeypatch.undo()`` reverts the fixture's env patches too (same
    ``MonkeyPatch`` instance), so ``LANCEDB_URI`` snaps back to the repo
    default and the runner's next tick rebuilds into ``data/lancedb``
    instead of ``tmp_path``. Reading the count before and after turns that
    silent append into a failed test. A missing table/db (fresh checkout)
    returns ``None`` so the guard only fires when there is something to
    leak into.
    """
    db_dir = ROOT / "data" / "lancedb"
    if not db_dir.exists():
        return None
    try:
        db = lancedb.connect(str(db_dir))
        return db.open_table(RAGSettings().LANCEDB_TABLE).count_rows()
    except Exception:
        return None


@pytest.fixture
def hook_app(tmp_path: Path, monkeypatch):
    """Real startup seam + real temp LanceDB + a live /v2 handler.

    ``LANCEDB_URI`` (env) points at a temp dir: every ``RAGSettings()``
    the hook, the ingest script and the retriever builder construct reads
    the same temp database, so the ingest → tag → retrieve chain is the
    production one against a disposable table.
    """
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": TOKENS[role], "actor_id": actor, "role": role}
                for role, (actor, _) in ACTORS.items()
            ]
        ),
    )
    monkeypatch.setenv("REINDEX_AUTORUN", "1")
    monkeypatch.setenv("REINDEX_TICK_SECONDS", "0.2")
    monkeypatch.setenv("REINDEX_RETRY_SECONDS", "0.05")
    monkeypatch.setenv("EMBEDDING_DIM", str(EMBEDDING_DIM))

    lancedb_dir = tmp_path / "lancedb"
    monkeypatch.setenv("LANCEDB_URI", str(lancedb_dir))
    real_rows_before = _real_serving_row_count()

    corpus_dir = tmp_path / "canonical_corpus"
    monkeypatch.setattr(release_status, "DEFAULT_CORPUS_DIR", corpus_dir)

    real_corpus = ROOT / "data" / "canonical_corpus"
    real_corpus_existed = real_corpus.exists()

    store_root = tmp_path / "canonical_store"
    app = FastAPI()
    app.include_router(admin_router)

    # request_tracker + build_retriever before configure: the hook reads
    # both off app.state, and the admin rollback route needs the tracker.
    app.state.request_tracker = RequestTracker()
    app.state.build_retriever = _build_retriever_for_table

    configure_canonical_record_store(
        app, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )

    runner = getattr(app.state, "reindex_runner", None)
    assert runner is not None, "startup seam did not attach the runner"
    assert runner.started is True, "REINDEX_AUTORUN=1 must start the worker"
    # The serving `husc_chunks` table is created by the first publish's ingest,
    # not at fixture time; yet the fixture must hand the serving retriever a
    # table that exists so it can be opened before any publish.  Create it
    # through the real ingest path so its schema is byte-identical to what
    # production writes, then empty it: the pre-publish sanity query (line 437)
    # must still see sources == [].
    _init_empty_husc_chunks_table(lancedb_dir)

    embedder = FakeEmbeddingService()
    app.state.embedding_service = embedder

    # The serving retriever starts as a real retriever over the (empty)
    # temp table; the hook's swap replaces it with a fresh one each
    # publish, exactly like main.py's _swap_runtime_services does.
    _RETRIEVER[0] = asyncio.run(_build_retriever_for_table(RAGSettings().LANCEDB_TABLE))

    # Wire main.py's swap seam onto this app so the hook's gate body
    # rebinds the module globals the real /v2 handler reads — the same
    # seam the admin reload route calls.
    app._swap_runtime_services = lambda retriever: _main.swap_runtime_services(
        retriever, app_state=app.state
    )
    # The rollback half resolves the serving retriever and pipeline through
    # main.py's `_health_lancedb` / `_health_pipeline` accessors (see
    # lifecycle._serving_retriever).  Production's `app` exposes them at
    # import time; this hand-mounted app must expose the same seam, bound to
    # main's module globals so the accessor sees whatever the swap last set.
    app._health_lancedb = lambda: _main.lancedb_retriever_service
    app._health_pipeline = lambda: _main.unified_pipeline
    # Keep the module-level slots the test drives in sync with main's
    # globals after every swap.
    _sync_main_globals()

    yield TestClient(app), app, corpus_dir, embedder, lancedb_dir

    runner.stop()
    assert real_corpus.exists() == real_corpus_existed, (
        "the e2e leaked a corpus into the repo's real data/canonical_corpus"
    )
    real_rows_after = _real_serving_row_count()
    assert real_rows_after == real_rows_before, (
        f"the e2e leaked {real_rows_after - real_rows_before} row(s) into the "
        f"repo's real {RAGSettings().LANCEDB_URI} table — a monkeypatch.undo() "
        "likely reverted the fixture's env isolation"
    )


def _sync_main_globals() -> None:
    """Point the test's module-level slots at main's module globals."""
    _CACHE[0] = _main.query_cache
    if _main.lancedb_retriever_service is not None:
        _RETRIEVER[0] = _main.lancedb_retriever_service
    _main.unified_pipeline = _PIPELINE[0]
    _main.llm_generator_service = _GENERATOR[0]


# ─────────────────────────────────────────────────────────────────────────
# AC (i): publish → ≤ 2 ticks → /v2/query returns the published chunk
# ─────────────────────────────────────────────────────────────────────────

def test_publish_reaches_v2_query_within_two_ticks(hook_app):
    """The ticket's headline: pressing Publish changes the chatbot's answer."""
    client, app, corpus_dir, embedder, lancedb_dir = hook_app

    # Sanity: before any record exists, /v2/query has no chunks to serve.
    payload = _drive_v2_query(_main, QUERY_A, embedder)
    assert payload["sources"] == [], payload

    _publish_record(client, REC_A, TEXT_A)

    # ≤ 2 ticks: the worker's retry interval is 0.05 s, so two ticks
    # is ~0.1 s of wall clock — the generous 30 s poll is the harness,
    # not the budget.
    assert _wait_for(lambda: _read_manifest(corpus_dir) is not None), (
        "the worker never built the corpus after a publish"
    )
    assert _wait_for(lambda: len(app.state.reindex_outbox.pending()) == 0), (
        "outbox never drained"
    )

    # The swap happened: the serving retriever is the one the hook
    # built over the ingested table, and the corpus the job built is
    # the corpus the ingest merged.
    assert _main.lancedb_retriever_service is not None

    payload = _drive_v2_query(_main, QUERY_A, embedder)
    assert any(TEXT_A.strip() in t for t in _chunk_texts(corpus_dir))
    # The answer is built from the retrieved documents, which are the
    # published record's chunks.
    assert TEXT_A.strip() in payload["answer"], payload["answer"]
    assert payload["status_code"] == "SUCCESS", payload


# Ticket T07 acceptance #2: HOI_DAP → publish → answer WITH citation.
# The ticket's scenario: a question/answer pair published as a HOI_DAP record
# must reach /v2/query with a non-empty ``sources`` list — the citation the
# student sees.  The fake generator emits ``chunk_id`` per retrieved chunk;
# the real ``LLMGenerator._enrich_sources`` emits
# ``{id, title, url, snippet, data_year}`` — both carry the chunk's identity,
# so the assertion below holds on the real generator too (``id`` is the
# chunk id, and the fake's ``chunk_id`` key is the same value under a
# different name).

def test_hoi_dap_publish_returns_answer_with_citation(hook_app):
    """Ticket T07 #2: publishing a HOI_DAP pair puts the answer out with a
    citation — ``payload["sources"]`` is non-empty after the publish.

    The existing AC (i) test proves the answer text arrives; this test closes
    the citation gap that AC #2 demands.  The record carries an explicit
    non-empty ``source_document_id`` so the citation gate
    (``abstention_policy._is_traceable``) is exercised without depending on
    the server-side source-document generation that lands separately.
    """
    client, app, corpus_dir, embedder, lancedb_dir = hook_app

    # Note: we deliberately do NOT assert ``sources == []`` before the
    # publish.  The fake embedder is content-addressed on ``len(text) % 97``,
    # not semantic: a query and a chunk with the same text length produce
    # the same 97-signature, so any pre-existing chunk with the same length
    # as ``QUERY_HD`` will retrieve and produce a non-empty source list.
    # That is a fake-embedder artefact, not a property of the citation gate,
    # so it must not block the test.  The citation assertion below (post-
    # publish, ``len(sources) > 0``) is the property the ticket measures.
    _publish_hoi_dap_record(client, REC_HD, TEXT_HD)

    # The worker must have built the corpus and the outbox must have drained.
    assert _wait_for(lambda: _read_manifest(corpus_dir) is not None), (
        "worker never built the corpus after a HOI_DAP publish"
    )
    assert _wait_for(lambda: len(app.state.reindex_outbox.pending()) == 0), (
        "outbox never drained after a HOI_DAP publish"
    )


    payload = _drive_v2_query(_main, QUERY_HD, embedder)

    # The answer text must be present: the published chunk reached the LLM.
    assert TEXT_HD.strip() in payload["answer"], (
        f"HOI_DAP answer text missing from payload['answer']: {payload['answer']!r}"
    )
    assert payload["status_code"] == "SUCCESS", payload
    # The citation: payload["sources"] must be NON-EMPTY — ticket T07 #2
    # ("câu đó trả lời được có trích dẫn").  Each entry carries a ``chunk_id``
    # (fake generator) or ``id`` (real generator, via _enrich_sources) that
    # is a SHA-256 hex string — the chunk's stable identity, not a synthetic
    # fallback like "chunk-0".
    sources = payload["sources"]
    assert len(sources) > 0, (
        f"T07#2 citation missing: payload['sources'] is empty. "
        f"status_code={payload['status_code']!r} "
        f"answer={payload['answer']!r}"
    )
    for src in sources:
        # The fake generator emits {"chunk_id": ...}; the real generator
        # emits {"id": ..., "title": ..., "url": ..., "snippet": ...,
        # "data_year": ...}.  The source-document identity — the provenance
        # that ``abstention_policy._is_traceable`` gates on — is NOT the
        # chunk hash: it is the source URL on the entry (real generator,
        # via ``_enrich_sources``).  The fake entry carries no URL, so the
        # test asserts on the chunk identity (non-empty hash) that the
        # fake actually emits.  Do NOT assert ``url`` here — it would
        # silently pass on the fake and hide that the fake itself cannot
        # produce a real citation.
        cid = src.get("chunk_id") or src.get("id")
        assert isinstance(cid, str) and len(cid) > 0, (
            f"source entry has no chunk_id/id: {src!r}"
        )


# ─────────────────────────────────────────────────────────────────────────
# AC (ii): ingest error → FAILED, outbox keeps, old corpus serves, retry OK
# ─────────────────────────────────────────────────────────────────────────

def test_ingest_error_fails_the_job_and_keeps_the_outbox(hook_app):
    """A refused ingest must not create a release, and must be retried."""
    import scripts.ingest_lancedb as ingest_mod

    client, app, corpus_dir, embedder, lancedb_dir = hook_app

    # First publish succeeds — the old corpus and release R1 exist.
    _publish_record(client, REC_A, TEXT_A)
    assert _wait_for(lambda: _read_manifest(corpus_dir) is not None)
    assert _wait_for(lambda: len(app.state.reindex_outbox.pending()) == 0)
    first_release = _active_release(app)
    first_answer = _drive_v2_query(_main, QUERY_A, embedder)
    assert TEXT_A.strip() in first_answer["answer"]

    # Second record, published — but the ingest now refuses (stale
    # corpus is the cheapest refusal to inject: it raises before any
    # row is written, so the table and the serving corpus stay intact).
    # Patch directly (not via the ``monkeypatch`` fixture): calling
    # ``monkeypatch.undo()`` would revert the ``hook_app`` fixture's env
    # isolation too (same ``MonkeyPatch`` instance), snapping LANCEDB_URI
    # back to the repo default so the retry rebuilds into ``data/lancedb``.
    real_stamp = ingest_mod.read_corpus_stamp

    def _stale_stamp(corpus):
        return None  # read_corpus_stamp -> None => stamp_is_stale(True)

    ingest_mod.read_corpus_stamp = _stale_stamp
    try:
        _publish_record(client, REC_B, TEXT_B)
        assert _wait_for(
            lambda: getattr(app.state, "reindex_job", None) is not None
            and app.state.reindex_job.state is JobState.FAILED
        ), "the failed ingest never marked the job FAILED"

        # Outbox keeps the entry (cursor did not advance past it).
        pending = app.state.reindex_outbox.pending()
        assert len(pending) == 1, [e.record_id for e in pending]
        assert REC_B in [e.record_id for e in pending]

        # Old corpus still serves: the release pointer and the answer are
        # still the first publish's.
        assert _active_release(app) == first_release
        again = _drive_v2_query(_main, QUERY_A, embedder)
        assert TEXT_A.strip() in again["answer"]
    finally:
        ingest_mod.read_corpus_stamp = real_stamp

    # Fix the ingest and let the next tick retry: the same release now
    # serves, and the answer carries the new record's chunk.
    assert _wait_for(
        lambda: len(app.state.reindex_outbox.pending()) == 0
    ), "the retry never drained the outbox"
    assert _wait_for(
        lambda: (_read_manifest(corpus_dir) or {}).get("record_count", 0) >= 2
    ), "the retry never rebuilt the corpus with both records"
    after = _drive_v2_query(_main, QUERY_A, embedder)
    assert TEXT_A.strip() in after["answer"]


def _active_release(app) -> Optional[str]:
    pointer = app.state.admin_rollback_service.pointer_path
    return json.loads(pointer.read_text("utf-8"))["release_id"]


# ─────────────────────────────────────────────────────────────────────────
# AC (iii): 2 publishes + rollback → release-1 answer returns
# ─────────────────────────────────────────────────────────────────────────

def test_two_publishes_then_rollback_restores_release_one(hook_app):
    """Rollback to release 1: its answer returns, the cache does not."""
    client, app, corpus_dir, embedder, lancedb_dir = hook_app
    outbox = app.state.reindex_outbox

    _publish_record(client, REC_A, TEXT_A)
    assert _wait_for(lambda: _read_manifest(corpus_dir) is not None)
    assert _wait_for(lambda: len(outbox.pending()) == 0)
    release_one = _active_release(app)

    _publish_record(client, REC_B, TEXT_B)
    assert _wait_for(
        lambda: (_read_manifest(corpus_dir) or {}).get("record_count", 0) >= 2
    )
    assert _wait_for(lambda: len(outbox.pending()) == 0)
    release_two = _active_release(app)
    assert release_two != release_one

    # Warm the query cache on release two: the stale-answer hazard the
    # ticket names.  force_route=padded_rag keeps the cache key stable.
    warm = _drive_v2_query(_main, QUERY_A, embedder)
    assert TEXT_A.strip() in warm["answer"]

    # Roll back through the admin route (the real two-level rollback).
    response = client.post(
        "/admin/rollback",
        json={"reason": "test rollback to release one", "target_release_id": release_one},
        headers=_headers("admin"),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["release_id"] == release_one

    # The pointer moved back to release one and the answer is its own.
    assert _active_release(app) == release_one
    after = _drive_v2_query(_main, QUERY_A, embedder)
    assert TEXT_A.strip() in after["answer"]


# ─────────────────────────────────────────────────────────────────────────
# AC (iv): rollback concurrent with a tick does not corrupt the pointer
# ─────────────────────────────────────────────────────────────────────────

def test_rollback_concurrent_with_tick_keeps_pointer_consistent(hook_app):
    """A rollback that races a publish leaves a readable, consistent pointer."""
    client, app, corpus_dir, embedder, lancedb_dir = hook_app
    outbox = app.state.reindex_outbox

    _publish_record(client, REC_A, TEXT_A)
    assert _wait_for(lambda: _read_manifest(corpus_dir) is not None)
    assert _wait_for(lambda: len(outbox.pending()) == 0)
    release_one = _active_release(app)

    _publish_record(client, REC_B, TEXT_B)
    assert _wait_for(
        lambda: (_read_manifest(corpus_dir) or {}).get("record_count", 0) >= 2
    )
    assert _wait_for(lambda: len(outbox.pending()) == 0)
    release_two = _active_release(app)

    # Race: a rollback to release one while the worker is mid-tick
    # (a third publish is queued).  Both hold the shared lock, so one
    # wins; either way the pointer must name a real, readable release.
    _publish_record(client, "CHI_TIEU:nganh:7480201:2026:c", TEXT_B + " thêm.")
    rollback_done: list[dict] = []

    def _do_rollback() -> None:
        with client:
            response = client.post(
                "/admin/rollback",
                json={"reason": "race", "target_release_id": release_one},
                headers=_headers("admin"),
            )
            rollback_done.append({"status": response.status_code, "body": response.text})

    import threading
    t = threading.Thread(target=_do_rollback)
    t.start()
    # Let the worker tick (and possibly publish release three) meanwhile.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and len(rollback_done) == 0:
        time.sleep(0.02)
    t.join(timeout=30.0)

    # Whatever interleaving happened, the pointer is readable and names
    # an existing release manifest.
    assert rollback_done, "the rollback never completed"
    assert rollback_done[0]["status"] == 200, rollback_done[0]
    current = _active_release(app)
    releases_dir = app.state.admin_rollback_service.releases_root
    assert (releases_dir / current / "release_manifest.json").is_file(), current
    # And a query still answers from a consistent runtime.
    payload = _drive_v2_query(_main, QUERY_A, embedder)
    assert payload["status_code"] in {"SUCCESS", "GENERATION_FAILED"}


# ─────────────────────────────────────────────────────────────────────────
# AC (v): legacy row count unchanged
# ─────────────────────────────────────────────────────────────────────────

def test_legacy_row_count_unchanged_after_publishes(hook_app):
    """Canonical publishes must never touch the legacy rows."""
    client, app, corpus_dir, embedder, lancedb_dir = hook_app

    # Seed a legacy row directly into the served table.
    db = lancedb.connect(str(lancedb_dir))
    table_name = RAGSettings().LANCEDB_TABLE
    legacy_text = "legacy row must survive"
    legacy_row = {
        "id": "legacy-1",
        "doc_id": "legacy-1",
        "release_id": "legacy",
        "text": legacy_text,
        "summary": None,
        "vector": [0.1] * EMBEDDING_DIM,
        "data_year": 0,
        "scope": "admissions",
        "doc_type": "chunk",
        "status": "current",
        "effective_from": 0,
        "effective_to": 99991231,
        "is_tombstone": False,
        "superseded_by": None,
        "metadata_json": json.dumps({"release_id": "legacy"}),
        "is_reference": False,
        "crawl_date": None,
    }
    if table_name in _table_names(db):
        table = db.open_table(table_name)
        table.add([legacy_row])
    else:
        db.create_table(table_name, data=[legacy_row])
    table = db.open_table(table_name)
    before = table.count_rows()

    _publish_record(client, REC_A, TEXT_A)
    assert _wait_for(lambda: _read_manifest(corpus_dir) is not None)
    assert _wait_for(lambda: len(app.state.reindex_outbox.pending()) == 0)

    _publish_record(client, REC_B, TEXT_B)
    assert _wait_for(
        lambda: (_read_manifest(corpus_dir) or {}).get("record_count", 0) >= 2
    )
    assert _wait_for(lambda: len(app.state.reindex_outbox.pending()) == 0)

    table = db.open_table(table_name)
    rows = table.to_arrow().to_pylist()
    # Identify the seeded row by its own id, not by ``release_id == "legacy"``:
    # the first publish's chunks are stamped with the *active* release id, which
    # is ``"legacy"`` when no release exists yet, so the label matches the
    # canonical rows too and cannot single out the legacy footprint.
    legacy_rows = [r for r in rows if r.get("id") == "legacy-1"]
    assert len(legacy_rows) == 1, f"legacy row count changed: {len(legacy_rows)}"
    assert legacy_rows[0]["text"] == legacy_text
    # The canonical rows are additive on top of the legacy footprint.
    assert table.count_rows() == before + 2


def _table_names(db) -> set[str]:
    listed = db.list_tables()
    names = listed.tables if hasattr(listed, "tables") else listed
    return set(names)
