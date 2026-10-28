"""T07: the publish hook closes the unanswered-queue loop at the successful
serving seam.

Two regression assertions the ticket requires:

* a publish whose ingest step raises must NOT resolve a queue item — the
  item stays open so a later retry or human publish can close it;
* a successful publish that serves a record carrying
  ``values.source_trace_id`` must resolve the matching open item, stamping
  ``resolved_by_record_id`` and ``resolved_at``.

The hook is the single production call site of
``UnansweredQueueService.resolve``; no other code path closes these items.

Notes on the test setup
-----------------------
``scripts`` is a namespace package (no ``__init__.py``).  The hook's
deferred ``from scripts.ingest_lancedb import ingest_canonical_corpus``
runs at call time; we pre-import the module so it is in ``sys.modules``
before the hook fires, then patch the attribute on that exact module
object via ``monkeypatch`` (which auto-restores after each test).

``FakeApp`` uses a real ``SimpleNamespace`` for ``state`` — not a
``MagicMock`` — so that ``getattr(app.state, "build_retriever", None)``
returns ``None`` and the hook's drain/swap branch is cleanly skipped,
letting the test focus on the T07 resolve step without a live LanceDB.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Pre-import scripts.ingest_lancedb so it lands in sys.modules before the
# hook's deferred import runs at call time.  The monkeypatch in each test
# then targets the exact module object the hook will see.
import scripts.ingest_lancedb  # noqa: F401  (registers in sys.modules)

from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.domain.canonical import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.services.unanswered_queue import UnansweredQueueService  # noqa: E402
from src.services.publish_hook import build_publish_hook  # noqa: E402
from datetime import date  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────────────

_TRACE_ID = "trace-t07-test"
_RECORD_ID = "HOI_DAP:quyet_dinh:1:2026:t07"


def _make_record_with_trace(
    record_id: str = _RECORD_ID,
    trace_id: str = _TRACE_ID,
) -> CanonicalRecord:
    """A minimal published HOI_DAP record carrying ``values.source_trace_id``."""
    return CanonicalRecord(
        record_id=record_id,
        record_type=RecordType.HOI_DAP,
        version=1,
        admission_year=2026,
        effective_from=date(2026, 1, 1),
        status=RecordStatus.PUBLISHED,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="doc-1",
        text="Hội đồng tuyển sinh xác nhận điểm chuẩn ngành CNTT năm 2026 là 24.0.",
        values={"source_trace_id": trace_id},
    )


def _make_chunks_jsonl(corpus_dir: Path, doc_id: str = _RECORD_ID) -> None:
    """Write a single-line chunks.jsonl that the hook will read."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    chunk = {
        "id": "test-chunk-001",
        "chunk_id": "test-chunk-001",
        "doc_id": doc_id,
        "text": "Điểm chuẩn ngành CNTT năm 2026 là 24.0.",
        "metadata": {"record_id": doc_id, "ingest_file": "chunks.jsonl"},
        "lineage": {},
    }
    (corpus_dir / "chunks.jsonl").write_text(
        json.dumps(chunk, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _make_vectors_jsonl(corpus_dir: Path) -> None:
    """Write a minimal vectors.jsonl so the ingest does not reject the corpus."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    vec = [0.0] * 4
    line = json.dumps({"chunk_id": "test-chunk-001", "vector": vec}, ensure_ascii=False)
    (corpus_dir / "vectors.jsonl").write_text(line + "\n", encoding="utf-8")


class FakeApp:
    """Minimal app carrying just the state the hook reads.

    ``state`` is a ``SimpleNamespace`` — every attribute the hook reads via
    ``getattr(app.state, ...)`` is ``None`` unless explicitly set, so the
    hook's drain/swap branch (``if builder is not None``) is cleanly skipped.
    """

    def __init__(
        self,
        queue: UnansweredQueueService,
        store: CanonicalRecordStore,
    ) -> None:
        self.state = SimpleNamespace(
            unanswered_queue=queue,
            canonical_record_store=store,
            build_retriever=None,
            request_tracker=None,
        )


def _build_hook(
    queue: UnansweredQueueService,
    store: CanonicalRecordStore,
    releases_root: Path,
) -> Any:
    app = FakeApp(queue, store)
    hook = build_publish_hook(
        app=app,
        releases_root=releases_root,
        releases_dir=releases_root / "releases",
        tracker=None,
        default_corp=releases_root / "corpus",
    )
    return hook


def _stage_corpus(tmp_path: Path, doc_id: str = _RECORD_ID) -> Path:
    """Build a minimal staged corpus dir the hook will copy to the serving dir."""
    staged = tmp_path / "staged"
    _make_chunks_jsonl(staged, doc_id)
    _make_vectors_jsonl(staged)
    # The hook copies these files to corpus_dir; the ingest is monkeypatched.
    return staged


def _patch_ingest_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``ingest_canonical_corpus`` on the exact module object in
    ``sys.modules`` so the hook's deferred import sees the patch."""
    import scripts.ingest_lancedb as ingest_mod

    monkeypatch.setattr(ingest_mod, "ingest_canonical_corpus", lambda corpus_dir: None)


def _patch_ingest_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.ingest_lancedb as ingest_mod

    def _fail_ingest(corpus_dir: Path) -> None:
        raise RuntimeError("stale corpus — ingest refused")

    monkeypatch.setattr(ingest_mod, "ingest_canonical_corpus", _fail_ingest)


# ── regression: ingest failure leaves queue item UNRESOLVED ─────────────────


def test_ingest_failure_does_not_resolve_queue_item(tmp_path: Path, monkeypatch) -> None:
    """When ingest_canonical_corpus raises, publish_hook raises too — the
    T07 step 5 is never reached, so a pre-existing open item stays open.
    """
    # 1. Set up store + queue with an open item for the trace.
    store_root = tmp_path / "store"
    store = CanonicalRecordStore(store_root)
    record = _make_record_with_trace()
    store.write_record(record)

    queue_root = tmp_path / "queue"
    queue = UnansweredQueueService(queue_path=queue_root / "queue.jsonl")
    queue.push("Câu hỏi về điểm chuẩn CNTT", _TRACE_ID, "served", "INSUFFICIENT_DATA")
    # Confirm the item is open before the hook runs.
    open_items = queue.unanswered()
    assert len(open_items) == 1
    assert open_items[0].trace_id == _TRACE_ID
    assert not open_items[0].resolved

    # 2. Build the hook; monkeypatch ingest to raise.
    _patch_ingest_fail(monkeypatch)

    hook = _build_hook(queue, store, tmp_path / "releases")
    staged = _stage_corpus(tmp_path)
    corpus_dir = tmp_path / "serving_corpus"

    # 3. The hook must raise (the job records FAILED); the item must stay open.
    with pytest.raises(Exception):
        hook(staged, corpus_dir)

    items_after = queue.items()
    assert len(items_after) == 1
    assert items_after[0].resolved is False, (
        "a failed publish must not close a queue item — the invariant is that "
        "resolution happens ONLY after ingest succeeded"
    )
    assert items_after[0].resolved_by_record_id is None


# ── positive: successful publish resolves the matching open item ─────────────


def test_successful_publish_resolves_matching_open_item(tmp_path: Path, monkeypatch) -> None:
    """A publish that ingests a record carrying ``values.source_trace_id``
    resolves the open queue item for that trace, stamping ``resolved_at``
    and ``resolved_by_record_id`` with the serving record's id.
    """
    # 1. Store + queue.
    store_root = tmp_path / "store"
    store = CanonicalRecordStore(store_root)
    record = _make_record_with_trace()
    store.write_record(record)

    queue_root = tmp_path / "queue"
    queue = UnansweredQueueService(queue_path=queue_root / "queue.jsonl")
    queue.push("Điểm chuẩn ngành CNTT 2026?", _TRACE_ID, "padded_rag", "INSUFFICIENT_DATA")
    assert len(queue.unanswered()) == 1

    # 2. Monkeypatch ingest to succeed (no LanceDB needed for the test).
    _patch_ingest_noop(monkeypatch)

    hook = _build_hook(queue, store, tmp_path / "releases")
    staged = _stage_corpus(tmp_path)
    corpus_dir = tmp_path / "serving_corpus"

    # 3. Run the hook — it must complete without raising.
    hook(staged, corpus_dir)

    # 4. The item is resolved, stamped with the serving record's id.
    items = queue.items()
    assert len(items) == 1
    resolved = items[0]
    assert resolved.trace_id == _TRACE_ID
    assert resolved.resolved is True, "a successful publish must resolve the open item"
    assert resolved.resolved_by_record_id == _RECORD_ID
    assert resolved.resolved_at is not None
    assert not queue.unanswered(), "no open items should remain after resolution"


# ── edge: no matching open item → nothing resolves, hook still succeeds ─────


def test_publish_with_unrelated_queue_item_leaves_it_open(tmp_path: Path, monkeypatch) -> None:
    """A queue item whose trace_id is not in the served corpus stays open."""
    store_root = tmp_path / "store"
    store = CanonicalRecordStore(store_root)
    record = _make_record_with_trace()  # trace-t07-test
    store.write_record(record)

    queue_root = tmp_path / "queue"
    queue = UnansweredQueueService(queue_path=queue_root / "queue.jsonl")
    unrelated_trace = "trace-unrelated"
    queue.push("Câu hỏi không liên quan", unrelated_trace, "guardrail", "NOT_IN_HUSC_SCOPE")

    _patch_ingest_noop(monkeypatch)

    hook = _build_hook(queue, store, tmp_path / "releases")
    staged = _stage_corpus(tmp_path)
    corpus_dir = tmp_path / "serving_corpus"

    hook(staged, corpus_dir)

    # The unrelated item is still open.
    open_items = queue.unanswered()
    assert len(open_items) == 1
    assert open_items[0].trace_id == unrelated_trace
    assert not open_items[0].resolved
