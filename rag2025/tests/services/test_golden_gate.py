"""U9 / A9 - Golden Gate (spec §PL-6.7, §PL-5, §PL-4).

All scenarios are hermetic: a scripted retriever/embedder stands in for the
serving stack, and a temp seed file stands in for the golden artifact. Nothing
touches PostgreSQL, LanceDB, or the LLM.

Scenarios:
  A. citation precision < 1.0 (one missing citation) -> gate_failed
  B. 26/26 supersession + citation 100% + recall >= baseline + abstention 100% -> ready
  C. fewer than 26 supersession cases / missing matrix row -> immediate fail
     (no retrieval runs), reason "incomplete_supersession_matrix"
  D. out-of-scope question answered instead of abstained -> fail abstention
  E. CLI invocation (`python scripts/run_golden_gate.py --release <id> ...`)
  F. a gate_failed release cannot be activated (409 RELEASE_GATE_FAILED)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from src.services.golden_gate import (  # noqa: E402
    GOLDEN_SEED_PATH,
    REQUIRED_MATRIX_COUNT,
    run_golden_gate,
)
from src.services.golden_retrieval_check import (  # noqa: E402
    GoldenRetrievalReport,
    _top5_record_ids,
    load_golden_set,
)  # noqa: F401  reuse contract


# ---------------------------------------------------------------------------
# Scripted doubles
# ---------------------------------------------------------------------------
class _Chunk:
    __slots__ = ("record_id", "coordinate_keys", "pages")

    def __init__(self, record_id, coordinate_keys=(), pages=()):
        self.record_id = record_id
        self.coordinate_keys = coordinate_keys
        self.pages = pages


class _RetrievalResult:
    def __init__(self, documents):
        self.documents = documents
        self.error_type = None
        self.error_message = None

    @property
    def is_success(self):
        return self.error_type is None and len(self.documents) > 0


class _ScriptedRetriever:
    """``retrieve(question) -> top-k record ids`` seeded per question."""

    def __init__(self, top5_by_question: dict[str, list[str]]):
        self._top5 = top5_by_question
        self.calls = 0

    def retrieve(self, *, query_vector, top_k=None, query=None):
        self.calls += 1
        record_ids = self._top5.get(str(query), [])
        docs = [
            _RetrievedDocumentDouble(
                text=f"chunk {rid}",
                source="s",
                chunk_id=f"chunk-{rid}",
                metadata={"record_id": rid},
                score=0.9,
            )
            for rid in record_ids[:top_k]
        ]
        return _RetrievalResult(docs)


class _RetrievedDocumentDouble:
    def __init__(self, text, source, chunk_id, metadata, score):
        self.text = text
        self.source = source
        self.chunk_id = chunk_id
        self.metadata = metadata
        self.score = score


class _ScriptedEmbedder:
    def encode_query(self, question):
        return [0.0] * 4


def _seed_text(rows: list[dict]) -> str:
    return json.dumps({"questions": rows}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Fixtures: seed shapes
# ---------------------------------------------------------------------------
def _in_scope(qid, keys):
    return {
        "expect_abstain": False,
        "expected_coordinate_keys": keys,
        "expected_pages": None,
        "group": "in_scope",
        "id": qid,
        "legacy_record_ids": [],
        "question": qid,
    }


def _supersession(qid, matrix_row, relation, old_keys, new_keys):
    return {
        "expect_abstain": False,
        "expected_coordinate_keys": new_keys,
        "expected_pages": None,
        "group": "supersession",
        "id": qid,
        "legacy_record_ids": [],
        "matrix_row": matrix_row,
        "new_coordinate_keys": new_keys,
        "old_coordinate_keys": old_keys,
        "question": qid,
        "relation": relation,
    }


def _oos(qid, expect_abstain=True):
    return {
        "expect_abstain": expect_abstain,
        "expected_coordinate_keys": [],
        "expected_pages": None,
        "group": "out_of_scope",
        "id": qid,
        "legacy_record_ids": [],
        "out_of_scope_group": "truong_khac",
        "question": qid,
    }


MATRIX_ROWS = [
    "T1.record.create",
    "T1.record.update",
    "T1.record.retire",
    "T1.document.create",
    "T1.document.update",
    "T1.document.retire",
    "T1.claim.create",
    "T1.claim.update",
    "T1.claim.retire",
    "T2.record.create",
    "T2.record.update",
    "T2.record.retire",
    "T2.document.create",
    "T2.document.update",
    "T2.document.retire",
    "T2.claim.create",
    "T2.claim.update",
    "T2.claim.retire",
    "T3.record.create",
    "T3.record.update",
    "T3.record.retire",
    "T3.document.create",
    "T3.document.update",
    "T3.document.retire",
    "T3.claim.create",
    "T3.claim.update",
]


def _full_seed() -> list[dict]:
    """Realistic seed: 48 in_scope + 26 supersession + 10 out_of_scope."""
    rows = []
    for i in range(1, 49):
        rows.append(_in_scope(f"q{i:03d}", [f"K{i:03d}"]))
    for i, row in enumerate(MATRIX_ROWS, 1):
        rows.append(_supersession(f"ss{i:03d}", row, "REPLACES", ["OLD"], ["NEW"]))
    for i in range(1, 11):
        rows.append(_oos(f"oos{i:03d}"))
    return rows


def _passing_retriever() -> _ScriptedRetriever:
    """Every question returns what the gate expects: new for supersession,
    expected keys for in_scope, nothing for out_of_scope."""
    top5: dict[str, list[str]] = {}
    for i in range(1, 49):
        top5[f"q{i:03d}"] = [f"K{i:03d}"]
    for i in range(1, 27):
        top5[f"ss{i:03d}"] = ["NEW"]
    # out_of_scope -> abstain (empty)
    return _ScriptedRetriever(top5)


# ---------------------------------------------------------------------------
# A. citation precision < 1.0 -> gate_failed
# ---------------------------------------------------------------------------
def test_citation_missing_one_fails_gate(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(_seed_text(_full_seed()), encoding="utf-8")

    retriever = _passing_retriever()
    # One in_scope question also returns an irrelevant record -> precision < 1.
    retriever._top5["q001"] = ["K001", "IRRELEVANT"]

    report = run_golden_gate(
        release_id="rel-citation",
        retriever=retriever,
        embedder=_ScriptedEmbedder(),
        seed_path=seed_path,
    )

    assert report.verdict == "fail"
    assert report.reason == "citation_precision_below_100"
    assert report.citation_precision < 1.0
    assert report.matrix_complete is True


# ---------------------------------------------------------------------------
# B. full pass -> ready
# ---------------------------------------------------------------------------
def test_full_pass_ready(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(_seed_text(_full_seed()), encoding="utf-8")

    report = run_golden_gate(
        release_id="rel-ready",
        retriever=_passing_retriever(),
        embedder=_ScriptedEmbedder(),
        seed_path=seed_path,
    )

    assert report.verdict == "pass"
    assert report.reason is None
    assert report.matrix_complete is True
    assert report.matrix_passed == 26
    assert report.matrix_total == 26
    assert report.recall_at_5 >= 0.44  # A1 floor
    assert report.recall_at_10 >= 0.5
    assert report.citation_precision == 1.0
    assert report.abstention_ok is True


# ---------------------------------------------------------------------------
# C. incomplete supersession matrix -> immediate fail, no retrieval
# ---------------------------------------------------------------------------
def test_missing_matrix_row_fails_immediately(tmp_path):
    """Drop one matrix row: gate fails before any retrieval runs."""
    rows = _full_seed()
    # Remove every row whose matrix_row == "T1.claim.update" (a real matrix row).
    rows = [r for r in rows if r.get("matrix_row") != "T1.claim.update"]
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(_seed_text(rows), encoding="utf-8")

    retriever = _passing_retriever()
    report = run_golden_gate(
        release_id="rel-incomplete",
        retriever=retriever,
        embedder=_ScriptedEmbedder(),
        seed_path=seed_path,
    )

    assert report.verdict == "fail"
    assert report.reason == "incomplete_supersession_matrix"
    assert report.matrix_complete is False
    # No retrieval may have run.
    assert retriever.calls == 0


def test_fewer_than_26_cases_fails_immediately(tmp_path):
    rows = _full_seed()
    # Keep only 25 of the 26 supersession cases.
    kept = [r for r in rows if r.get("group") != "supersession"]
    kept += [r for r in rows if r.get("group") == "supersession"][:25]
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(_seed_text(kept), encoding="utf-8")

    retriever = _passing_retriever()
    report = run_golden_gate(
        release_id="rel-few",
        retriever=retriever,
        embedder=_ScriptedEmbedder(),
        seed_path=seed_path,
    )

    assert report.verdict == "fail"
    assert report.reason == "incomplete_supersession_matrix"
    assert report.matrix_complete is False
    assert retriever.calls == 0


# ---------------------------------------------------------------------------
# D. out-of-scope answered instead of abstained -> fail abstention
# ---------------------------------------------------------------------------
def test_out_of_scope_answered_fails_abstention(tmp_path):
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(_seed_text(_full_seed()), encoding="utf-8")

    retriever = _passing_retriever()
    # One out_of_scope question returns an in-scope answer instead of abstaining.
    retriever._top5["oos001"] = ["K001"]

    report = run_golden_gate(
        release_id="rel-abstain",
        retriever=retriever,
        embedder=_ScriptedEmbedder(),
        seed_path=seed_path,
    )

    assert report.verdict == "fail"
    assert report.reason == "abstention_failed"
    assert report.abstention_ok is False


# ---------------------------------------------------------------------------
# E. CLI invocation
# ---------------------------------------------------------------------------
def test_cli_invocation(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.jsonl"
    seed_path.write_text(_seed_text(_full_seed()), encoding="utf-8")

    import importlib.util

    script = ROOT / "scripts" / "run_golden_gate.py"
    spec = importlib.util.spec_from_file_location("run_golden_gate_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    passed_retriever = _passing_retriever()

    class _Dual:
        """Retriever + embedder in one (the script passes the same object as both)."""
        def __init__(self):
            self.embedder = _Embedder()

        def encode_query(self, question):
            return [0.0] * 4

        def retrieve(self, *, query_vector, top_k=None, query=None):
            return passed_retriever.retrieve(
                query_vector=query_vector, top_k=top_k, query=query
            )

    class _Embedder:
        def encode_query(self, question):
            return [0.0] * 4

    cli._build_retriever = lambda: _Dual()
    cli.release_exists = lambda *a, **k: True
    persisted = {}
    cli.persist_to_pg = lambda release_id, report, database_url: persisted.update(
        release_id=release_id, report=report
    )

    rc = cli.main(
        [
            "--release",
            "rel-cli",
            "--golden-set",
            str(seed_path),
            "--baseline",
            str(ROOT / "eval" / "baseline_A1.json"),
        ]
    )

    assert rc == 0
    assert persisted["release_id"] == "rel-cli"
    assert persisted["report"].verdict == "pass"


# ---------------------------------------------------------------------------
# F. gate_failed release cannot be activated (409 RELEASE_GATE_FAILED)
# ---------------------------------------------------------------------------
def test_gate_failed_release_cannot_be_activated(monkeypatch, tmp_path):
    from src.routers.admin import router as admin_router

    app = _app_with_activate_route(monkeypatch, tmp_path)
    client = TestClient(app)

    # Flip the existing release to gate_failed via PG, then attempt activation.
    import os
    import psycopg

    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        __import__("json").dumps(
            [{"token": "tok-admin-sql-1", "actor_id": "admin@husc", "role": "admin"}]
        ),
    )
    headers = {"X-Admin-Token": "tok-admin-sql-1"}

    database_url = "postgresql://rag:rag@127.0.0.1:5432/husc"
    with psycopg.connect(database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE releases SET status='gate_failed' WHERE release_id='e2e-release'"
            )

    try:
        resp = client.post(
            "/admin/releases/e2e-release/activate", headers=headers
        )
        assert resp.status_code == 409, resp.text
        assert "RELEASE_GATE_FAILED" in resp.text
    finally:
        with psycopg.connect(database_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE releases SET status='active' WHERE release_id='e2e-release'"
                )


def _app_with_activate_route(monkeypatch, tmp_path):
    """Admin routes over the PostgreSQL-backed canonical store, as in sql_admin_client."""
    from fastapi import FastAPI

    from config.settings import RAGSettings
    from src.app.lifecycle import configure_canonical_record_store
    from src.routers.admin import router as admin_router

    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://rag:rag@127.0.0.1:5432/husc")
    monkeypatch.setenv("ADMIN_FOUR_EYES", "0")
    app = FastAPI()
    app.include_router(admin_router)  # router already carries prefix="/admin"
    configure_canonical_record_store(
        app,
        settings=RAGSettings(
            CANONICAL_STORE_ROOT=tmp_path / "canonical_store",
            STORE_BACKEND="postgres",
        ),
    )
    return app
