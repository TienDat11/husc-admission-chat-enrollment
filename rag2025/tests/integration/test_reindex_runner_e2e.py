"""Ticket 15 criterion 7, measured end-to-end: production RUNS the job.

The audit's B15-1 was precise: ``ReindexJob`` existed, the outbox recorded
every publication, the admin screen reported the queue - and nothing
connected them, so the corpus was rebuilt only when a human remembered a
script. This file measures the opposite claim, at the HTTP seam, with the
real wiring:

1. publish a record through ``POST /admin/records/.../transitions`` ->
   the runner's own tick drains the outbox, materializes the release,
   builds the corpus, advances the cursor, and ``GET /admin/index-jobs``
   reports the job ``active`` - with no manual reindex command anywhere;
2. publish a second record -> the next tick rebuilds INCREMENTALLY, and
   the fake embedder sees only the new record's chunks (the old ones are
   reused by content hash);
3. bump the chunker version stamp (monkeypatch ``version_stamp``) -> the
   next tick performs a FULL rebuild on its own. This is the trigger a
   record-level queue can never see, and the test proves nobody typed a
   command between the bump and the rebuilt corpus.

The app is assembled the way ``tests/routers/test_admin_records_api.py``
assembles it - a fresh ``FastAPI`` with the admin router and the real
``configure_canonical_record_store`` startup seam - so the runner, the
outbox, the lifecycle service and the read model are the same objects
production wires, over a tmp store and a tmp corpus.

Two guards keep the suite honest:

* the worker is started by the wiring (``REINDEX_AUTORUN=1`` set before
  ``configure``), and every test stops it in fixture teardown - before
  monkeypatched stamps/env revert - so no daemon outlives its test;
* the repo's real ``data/canonical_corpus`` must not appear: the corpus
  dir is rebound to ``tmp_path`` at the same module global the runner and
  the screen read, and the teardown asserts the real path is untouched.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from rag2025.src.app.lifecycle import (  # noqa: E402
    configure_canonical_record_store,
)
from src.routers.admin import router as admin_router  # noqa: E402
from src.services import reindex_job as reindex_job_module  # noqa: E402
from src.services import release_status  # noqa: E402
from src.services.canonical_chunk_builder import (  # noqa: E402
    CHUNKS_FILENAME,
    CHUNK_MANIFEST_FILENAME,
)

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

#: The literal the stamp bump writes. The phase-3 assertions are about this
#: string appearing in the corpus manifest with nobody having run a command.
BUMPED_CHUNKER = "chunker-e2e-bumped-v99"


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _record_body(record_id: str) -> dict:
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
        "text": f"Chỉ tiêu tuyển sinh 2026 cho ngành {record_id.split(':')[2]}.",
    }


class FakeEmbeddingService:
    """Stand-in for ``EmbeddingService`` at the ``encode_documents`` seam.

    The runner resolves ``getattr(service, "encode_documents", service)`` per
    tick, so this object is what ``app.state.embedding_service`` holds in
    production - the same attribute, the same call shape. Vectors are
    deterministic in the text so content-hash reuse behaves identically to a
    real model on unchanged input.
    """

    EMBEDDING_DIM: int = 1024

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.batches: list[list[str]] = []

    def encode_documents(self, documents):
        texts = list(documents)
        self.batches.append(texts)
        return [
            [float(len(text) % 97), *([0.5] * (self.dim - 1))] for text in texts
        ]

    @property
    def embedded_texts(self) -> list[str]:
        return [text for batch in self.batches for text in batch]


def _wait_for(predicate, timeout: float = 30.0, interval: float = 0.02) -> bool:
    """Poll until ``predicate`` holds - the worker owns the timing, not the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _real_lancedb_row_count() -> int | None:
    """Rows in the repo's REAL serving table, or ``None`` when it is absent.

    The leak this guards against was invisible: the fixture's publish hook
    ingested into ``data/lancedb`` because ``LANCEDB_URI`` was never rebound.
    Reading the count before and after turns a silent append into a failed
    test. A missing table/db (fresh checkout) returns ``None`` so the guard
    only fires when there is something to leak into.
    """
    db_dir = ROOT / "data" / "lancedb"
    if not db_dir.exists():
        return None
    try:
        import lancedb  # noqa: PLC0415 - optional heavy import, test-only

        db = lancedb.connect(str(db_dir))
        return db.open_table(RAGSettings().LANCEDB_TABLE).count_rows()
    except Exception:
        return None


def _read_manifest(corpus_dir: Path) -> dict | None:
    """The serving corpus manifest, or ``None`` while it is absent/unwritten.

    A mid-swap read can see a half-written file; ``None`` makes the poller
    retry instead of failing on a torn read.
    """
    try:
        return json.loads((corpus_dir / CHUNK_MANIFEST_FILENAME).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _chunk_texts_by_record(corpus_dir: Path) -> dict[str, set[str]]:
    """record_id -> the chunk texts the corpus holds for it."""
    by_record: dict[str, set[str]] = {}
    lines = (corpus_dir / CHUNKS_FILENAME).read_text("utf-8").splitlines()
    for line in lines:
        chunk = json.loads(line)
        by_record.setdefault(chunk["metadata"]["record_id"], set()).add(chunk["text"])
    return by_record


def _publish(client: TestClient, record_id: str) -> None:
    """Create a draft and walk it to published over HTTP, like the UI would."""
    created = client.post(
        "/admin/records",
        json={"record": _record_body(record_id)},
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


@pytest.fixture
def runner_app(tmp_path: Path, monkeypatch):
    """The real startup seam, autorun on, corpus rebound to tmp. Teardown stops the worker."""
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": TOKENS[role], "actor_id": actor, "role": role}
                for role, (actor, _) in ACTORS.items()
            ]
        ),
    )
    # Criterion 7 says the worker runs in production; the gate's pytest default
    # is off, so the e2e turns it on explicitly - the ticket's "set tường minh
    # trong test e2e" - and shortens both sleeps so ticks are fast, not absent.
    monkeypatch.setenv("REINDEX_AUTORUN", "1")
    monkeypatch.setenv("REINDEX_TICK_SECONDS", "0.2")
    monkeypatch.setenv("REINDEX_RETRY_SECONDS", "0.05")

    # The runner resolves the corpus dir lazily off this module global, and
    # IndexJobsService binds it per request - one patch moves both, so the
    # worker and the screen can never disagree about which corpus is "the"
    # corpus, and neither can touch the repo's real data/canonical_corpus.
    corpus_dir = tmp_path / "canonical_corpus"
    monkeypatch.setattr(release_status, "DEFAULT_CORPUS_DIR", corpus_dir)

    # LANCEDB_URI is rebound for the same reason the corpus dir is: the publish
    # hook's ingest writes into whatever database the settings name, and an
    # unpatched env appends the test record to the repo's REAL data/lancedb
    # (measured: one stray release row per run, table 1679 -> 1680). Every
    # RAGSettings() the hook and the ingest script build now reads this temp
    # database, so the chain is production's against a disposable table.
    lancedb_dir = tmp_path / "lancedb"
    monkeypatch.setenv("LANCEDB_URI", str(lancedb_dir))
    monkeypatch.setenv("EMBEDDING_DIM", str(FakeEmbeddingService.EMBEDDING_DIM))

    real_corpus = ROOT / "data" / "canonical_corpus"
    real_corpus_existed = real_corpus.exists()
    real_lancedb_rows = _real_lancedb_row_count()

    store_root = tmp_path / "canonical_store"
    app = FastAPI()
    app.include_router(admin_router)
    configure_canonical_record_store(
        app, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )

    # The wiring starts the worker before any embedder exists - the real
    # process assigns ``app.state.embedding_service`` later too - so the
    # early ticks must abstain, not crash.
    runner = getattr(app.state, "reindex_runner", None)
    assert runner is not None, "startup seam did not attach the runner"
    assert runner.started is True, "REINDEX_AUTORUN=1 must start the worker"

    embedder = FakeEmbeddingService()
    app.state.embedding_service = embedder

    yield TestClient(app), app, corpus_dir, embedder

    # Teardown order matters: this fixture depends on monkeypatch, so it is
    # torn down FIRST - the worker stops before the stamp/env patches revert.
    runner.stop()
    assert real_corpus.exists() == real_corpus_existed, (
        "the e2e leaked a corpus into the repo's real data/canonical_corpus"
    )
    assert _real_lancedb_row_count() == real_lancedb_rows, (
        "the e2e leaked rows into the repo's real data/lancedb husc_chunks table"
    )


def test_publish_drains_the_outbox_and_the_screen_reports_active(runner_app):
    """Criterion 7, direction one: a publish alone produces a serving corpus.

    Nothing in this test calls ``run_once``, constructs a ``ReindexJob``, or
    shells out to the CLI. The only writes are the HTTP publish, and the
    corpus appears because the worker decided to build it.
    """
    client, app, corpus_dir, embedder = runner_app
    outbox = app.state.reindex_outbox

    _publish(client, REC_A)

    assert _wait_for(lambda: _read_manifest(corpus_dir) is not None), (
        "the worker never built the corpus after a publish"
    )
    manifest = _read_manifest(corpus_dir)
    # First build: no corpus stated a stamp, so the plan is a FULL rebuild -
    # and the outbox entry is what made it due at all.
    assert manifest["mode"] == "full"
    assert REC_A in manifest["affected_record_ids"]
    assert (corpus_dir / CHUNKS_FILENAME).exists()

    assert _wait_for(lambda: len(outbox.pending()) == 0), "outbox never drained"
    assert outbox.cursor() >= 1, "the cursor never advanced past the entry"

    # The record-level trigger re-embedded exactly the new record's chunks.
    assert embedder.batches, "the fake embedder was never called"
    embedded = set(embedder.embedded_texts)
    assert embedded == _chunk_texts_by_record(corpus_dir)[REC_A]

    body = client.get("/admin/index-jobs", headers=_headers("viewer")).json()
    live = [job for job in body["jobs"] if job["serving"] is True]
    assert live, f"index-jobs never reported a serving job: {body['jobs']}"
    assert live[0]["state"] == "active"
    assert body["pending_outbox_entries"] == 0
    assert body["reindex_needed"] is False


def test_second_publish_rebuilds_incrementally_with_only_its_chunks(runner_app):
    """The queue trigger alone must not re-embed what did not change.

    After the first corpus exists and its stamp matches the code, a second
    publish is due for exactly one reason: a pending entry. The plan is
    INCREMENTAL and the embedder sees only the new record's texts.
    """
    client, app, corpus_dir, embedder = runner_app
    outbox = app.state.reindex_outbox

    _publish(client, REC_A)
    assert _wait_for(
        lambda: (_read_manifest(corpus_dir) or {}).get("mode") == "full"
        and len(outbox.pending()) == 0
    )
    embedder.batches.clear()

    _publish(client, REC_B)
    assert _wait_for(
        lambda: (_read_manifest(corpus_dir) or {}).get("mode") == "incremental"
    ), "the second publish never produced an incremental rebuild"
    assert _wait_for(lambda: len(outbox.pending()) == 0)

    manifest = _read_manifest(corpus_dir)
    assert manifest["affected_record_ids"] == [REC_B]
    by_record = _chunk_texts_by_record(corpus_dir)
    embedded = set(embedder.embedded_texts)
    # Only the new record's chunks were embedded; the first record's were
    # reused by content hash - "chi tiết thay đổi được embed lại", measured.
    assert embedded == by_record[REC_B]
    assert not (embedded & by_record[REC_A])
    # Both records are still in the serving corpus.
    assert set(by_record) == {REC_A, REC_B}


def test_bumping_the_version_stamp_rebuilds_full_with_no_manual_command(
    runner_app, monkeypatch
):
    """Criterion 7, direction two - the trigger the queue cannot see.

    A chunker change produces no outbox entry: no record was republished.
    The only signal is the version stamp, and the ticket's claim is that the
    rebuild happens "mà không cần ai nhớ". So this test republishes nothing,
    runs no CLI, calls no ``run_once``: it patches the stamp and waits. If
    the runner's due-condition ignores staleness, the manifest never changes
    and this test times out red - which is exactly what M15b demonstrated.
    """
    client, app, corpus_dir, embedder = runner_app
    outbox = app.state.reindex_outbox

    _publish(client, REC_A)
    assert _wait_for(
        lambda: (_read_manifest(corpus_dir) or {}).get("mode") == "full"
        and len(outbox.pending()) == 0
    )
    before = _read_manifest(corpus_dir)
    cursor_before = outbox.cursor()
    embedder.batches.clear()

    # The bump: what the code would now call its version, changed under the
    # corpus. Patched where the runner's due-check and the job's manifest
    # both read it - ``CorpusStamp.current()`` and ``_manifest`` resolve this
    # one module global.
    real_stamp = reindex_job_module.version_stamp()

    def bumped_stamp():
        return {**real_stamp, "chunker_version": BUMPED_CHUNKER}

    monkeypatch.setattr(reindex_job_module, "version_stamp", bumped_stamp)
    assert _wait_for(
        lambda: (_read_manifest(corpus_dir) or {}).get("chunker_version")
        == BUMPED_CHUNKER
    ), "a stamp bump never triggered an automatic rebuild"

    manifest = _read_manifest(corpus_dir)
    assert manifest["mode"] == "full", (
        "a stale corpus must force a FULL rebuild, not an incremental one"
    )
    assert manifest["reason"].startswith("stamp-mismatch")
    # No publish happened after phase one: the queue stayed drained and the
    # cursor never moved. The rebuild was the stamp's doing alone.
    assert len(outbox.pending()) == 0
    assert outbox.cursor() == cursor_before
    assert manifest["built_at"] != before["built_at"]

    # And the screen still reports a serving job after the automatic run.
    # The manifest lands during the hand-over, one step before the job
    # transitions itself to ``active``; poll the screen so the assertion is
    # about the settled state, not about the tick's interleaving.
    assert _wait_for(
        lambda: any(
            job["serving"] is True
            for job in client.get(
                "/admin/index-jobs", headers=_headers("viewer")
            ).json()["jobs"]
        )
    ), "index-jobs never reported a serving job after the stamp bump"
    body = client.get("/admin/index-jobs", headers=_headers("viewer")).json()
    serving = [job for job in body["jobs"] if job["serving"] is True]
    assert serving[0]["state"] == "active"
    assert body["reindex_needed"] is False
