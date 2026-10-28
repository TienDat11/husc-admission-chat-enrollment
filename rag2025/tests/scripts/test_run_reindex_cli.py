"""The operator CLI over the production re-index runner - ticket 15 (criterion 7, manual path).

``scripts/run_reindex.py`` is the one-shot sibling of the background worker: it
builds a ``ReindexRunner`` over the configured store/outbox/embedder and calls
``run_once()`` exactly once. These tests drive ``main()`` in-process with a
fake embedder injected at the CLI's ``EmbeddingService`` seam, so every
assertion runs the real wiring: a publish through the admin lifecycle appends
an outbox entry, the CLI's run drains it, and the corpus on disk is the job's
actual output.

What is pinned here is the CLI's own contract — argument-to-path resolution,
outcome-to-exit-code mapping, and the loud-failure rules — not the runner's
planning (``tests/services/test_reindex_runner.py``) or the job's building
(``tests/services/test_reindex_job.py``).
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.domain.canonical.lifecycle import LifecycleAction  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services.admin_lifecycle import AdminLifecycleService  # noqa: E402
from src.services.reindex_outbox import OUTBOX_FILENAME, ReindexOutbox  # noqa: E402

import scripts.run_reindex as run_reindex  # noqa: E402

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
NOW = datetime(2026, 9, 13, 10, 0, tzinfo=timezone.utc)


class FakeEmbeddingService:
    """Deterministic stand-in for the real lazy model: same ``encode_documents``
    surface, no weights. Vectors are content-derived so two identical chunks
    embed identically and different chunks do not."""

    def __init__(self, settings) -> None:
        self.settings = settings

    def encode_documents(self, documents):
        return np.array(
            [[float(sum(b for b in doc.encode("utf-8")) % 977) / 977.0, 1.0] for doc in documents],
            dtype=np.float32,
        )


def _record(text: str = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm.") -> CanonicalRecord:
    return CanonicalRecord(
        record_id=RECORD_ID,
        record_type=RecordType.DIEM_CHUAN,
        version=1,
        admission_year=2026,
        effective_from="2026-06-01",
        status=RecordStatus.DRAFT,
        authority_level=AuthorityLevel.HUSC_OFFICIAL,
        source_document_id="de-an-tuyen-sinh-2026",
        text=text,
    )


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    return tmp_path / "canonical_corpus"


@pytest.fixture
def wired(tmp_path: Path, store_root: Path):
    """A store + outbox pair under one root, as the app wires them."""
    outbox = ReindexOutbox(store_root / "outbox" / OUTBOX_FILENAME)
    store = CanonicalRecordStore(store_root)
    service = AdminLifecycleService(store, outbox=outbox, clock=lambda: NOW)
    return store, outbox, service


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    """The CLI resolves its embedder through its own module attribute."""
    monkeypatch.setattr(run_reindex, "EmbeddingService", FakeEmbeddingService)


def _publish(service: AdminLifecycleService) -> CanonicalRecord:
    service.create(_record(), actor_id="cb.bien-tap", role="editor")
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="cb.phan-bien")
    return service.transition(RECORD_ID, LifecycleAction.PUBLISH, actor_id="cb.thu-ky")


def test_a_publish_then_one_cli_run_leaves_a_serving_corpus(
    store_root: Path, corpus_dir: Path, wired
):
    """The operator path end-to-end: publish -> outbox entry -> CLI -> corpus.

    This is criterion 7's "end-to-end, not by reading code" for the manual
    trigger: nothing here constructs a ReindexJob by hand; the CLI's exit
    code 0 and the bytes on disk are the runner's own doing.
    """
    _store, outbox, service = wired
    _publish(service)
    assert len(outbox.pending()) == 1, "the publish must queue exactly one trigger"

    code = run_reindex.main(
        ["--store-root", str(store_root), "--corpus-dir", str(corpus_dir)]
    )

    assert code == 0
    chunks = [
        json.loads(line)
        for line in (corpus_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert chunks, "the CLI run must leave a non-empty corpus"
    assert any(c["metadata"]["record_id"] == RECORD_ID for c in chunks)
    assert outbox.cursor() == len(outbox.entries()), "the drain must move the cursor"
    assert outbox.pending() == ()


def test_a_second_cli_run_with_nothing_pending_reports_nothing_due(
    store_root: Path, corpus_dir: Path, wired, capsys
):
    """Idempotent operator path: run again, nothing is re-embedded, exit 0."""
    _store, _outbox, service = wired
    _publish(service)
    argv = ["--store-root", str(store_root), "--corpus-dir", str(corpus_dir)]
    assert run_reindex.main(argv) == 0
    before = (corpus_dir / "chunks.jsonl").read_bytes()

    capsys.readouterr()
    code = run_reindex.main(argv)
    out = capsys.readouterr().out

    assert code == 0
    assert "nothing due" in out, out
    assert (corpus_dir / "chunks.jsonl").read_bytes() == before


def test_a_stale_corpus_is_rebuilt_by_a_bare_cli_run(
    store_root: Path, corpus_dir: Path, wired, monkeypatch, capsys
):
    """A stamp change re-triggers the full path with no extra argument.

    The operator types the same command as before; the runner notices the
    corpus was built by different logic and rebuilds it, which pins that the
    CLI honours the stamp trigger without a ``--force``-style flag.
    """
    _store, _outbox, service = wired
    _publish(service)
    argv = ["--store-root", str(store_root), "--corpus-dir", str(corpus_dir)]
    assert run_reindex.main(argv) == 0
    manifest_path = corpus_dir / "chunk_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Pretend the chunking logic moved under the corpus (what ticket 32/33's
    # real CHUNKER_VERSION bump does), leaving the outbox fully drained.
    # The stamp lives at the manifest's TOP level (CorpusStamp.from_manifest
    # reads the flat file), so the stale value goes there.
    assert manifest["chunker_version"], "a built corpus must state its stamp"
    manifest_path.write_text(
        json.dumps({**manifest, "chunker_version": "0.0-test-stale"},
                   ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    capsys.readouterr()

    code = run_reindex.main(argv)
    out = capsys.readouterr().out

    assert code == 0, out
    assert "mode=full" in out, out
    rebuilt = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert rebuilt["chunker_version"] != "0.0-test-stale"


def test_a_failing_run_exits_nonzero_and_leaves_the_serving_corpus_untouched(
    store_root: Path, corpus_dir: Path, wired, monkeypatch
):
    """A broken embedder must not cost the deployment its current index."""
    _store, _outbox, service = wired
    _publish(service)
    argv = ["--store-root", str(store_root), "--corpus-dir", str(corpus_dir)]
    assert run_reindex.main(argv) == 0
    before = {
        p.name: p.read_bytes() for p in corpus_dir.glob("*") if p.is_file()
    }

    class BrokenEmbeddingService(FakeEmbeddingService):
        def encode_documents(self, documents):
            raise RuntimeError("embedding backend unavailable")

    monkeypatch.setattr(run_reindex, "EmbeddingService", BrokenEmbeddingService)
    # A second publication so the next run is due and reaches the embedder:
    # v1 is immutable once published, so the new text goes in as version 2.
    service.edit(
        RECORD_ID,
        text="Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 26,0 điểm.",
        actor_id="cb.bien-tap",
    )
    service.transition(RECORD_ID, LifecycleAction.SUBMIT, actor_id="cb.bien-tap")
    service.transition(RECORD_ID, LifecycleAction.APPROVE, actor_id="cb.phan-bien")
    service.transition(RECORD_ID, LifecycleAction.PUBLISH, actor_id="cb.thu-ky")

    code = run_reindex.main(argv)

    assert code == 1
    assert {
        p.name: p.read_bytes() for p in corpus_dir.glob("*") if p.is_file()
    } == before


def test_an_unbuildable_runner_exits_two(tmp_path: Path, monkeypatch):
    """Configuration errors are exit 2, distinct from a failed run (exit 1)."""

    class ExplodingStore:
        def __init__(self, root: Path) -> None:
            raise OSError(13, "permission denied")

    monkeypatch.setattr(run_reindex, "EmbeddingService", FakeEmbeddingService)
    import src.infrastructure.canonical_record_store as crs_mod

    monkeypatch.setattr(crs_mod, "CanonicalRecordStore", ExplodingStore)
    code = run_reindex.main(["--store-root", str(tmp_path / "nope")])
    assert code == 2


def test_a_skipped_run_exits_one(store_root: Path, monkeypatch, capsys):
    """Work was due but the runner could not start it: exit 1, on stderr.

    The skipped branch is the one the real filesystem cannot produce on
    command (the CLI always has an embedder), so the runner object itself is
    stubbed at the CLI's construction seam. A cron wrapper must not read
    "due but skipped" as success.
    """
    from src.services.reindex_runner import RunnerOutcome

    class SkippingRunner:
        def __init__(self, **kwargs) -> None:
            pass

        def run_once(self):
            return RunnerOutcome(
                due=True, action="skipped", reason="no published records to index"
            )

    monkeypatch.setattr(run_reindex, "ReindexRunner", SkippingRunner)
    capsys.readouterr()

    code = run_reindex.main(["--store-root", str(store_root)])
    captured = capsys.readouterr()

    assert code == 1
    assert "skipped" in captured.err, captured.err
