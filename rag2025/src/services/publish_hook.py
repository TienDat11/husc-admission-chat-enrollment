"""Publish hook — ticket T01: connect the reindex hand-over to the serving path.

When a publish lands in the outbox and the runner rebuilds the corpus, this
module supplies the ``publish`` callable that ``ReindexJob._hand_over`` calls
instead of the default file swap. The hook does the full publish in one pass:

1. copy the staged corpus into the serving corpus directory (default publish);
2. ingest the corpus into LanceDB (merge_insert, no legacy rows touched);
3. build the release: id, tag, manifest with parent, pointer, graph pin;
4. drain-gate the runtime swap under the shared ``_reload_lock``;
5. clear the query cache so stale answers are not served from the new release.

The hook runs in the runner's worker thread, not the event loop.  Step 4 is
async (the drain gate), so it is scheduled onto the app's event loop via
``asyncio.run_coroutine_threadsafe`` when one is available; in the TestClient
case (no live loop at call time) the gate is run inline in a fresh loop, which
is safe because no real in-flight traffic exists.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from src.domain.release import ReleaseManifest
from src.infrastructure.canonical_store import publish_pointer, write_manifest
from src.infrastructure.lancedb_versioning import tag_release
from src.services.admin_rollback import (
    MANIFEST_FILENAME,
    POINTER_FILENAME,
    RELEASES_DIRNAME,
)
from src.services.query_cache import QueryCache

#: Default drain timeout, same as the admin reload route's F4 budget.
_PUBLISH_DRAIN_TIMEOUT_S = 25.0


# ---------------------------------------------------------------------------
# Release-manifest factory
# ---------------------------------------------------------------------------

def make_release_manifest(
    *,
    release_id: str,
    corpus_dir: Path,
    parent_release_id: Optional[str] = None,
) -> ReleaseManifest:
    """Build a ``ReleaseManifest`` for a freshly ingested corpus.

    The chunk count is read off the corpus manifest the job just wrote, so a
    release cannot claim a footprint it did not build. The manifest's
    ``lancedb_table``/``lancedb_tag``/``lancedb_version`` are filled in after
    the table is tagged; the caller patches them with ``replace`` before writing.
    """
    chunk_count = 0
    try:
        manifest_path = Path(corpus_dir) / "chunk_manifest.json"
        if manifest_path.is_file():
            chunk_count = int(
                json.loads(manifest_path.read_text(encoding="utf-8")).get(
                    "chunk_count", 0
                )
            )
    except (OSError, ValueError, TypeError):
        chunk_count = 0
    return ReleaseManifest(
        release_id=release_id,
        created_at=datetime.now(timezone.utc),
        parent_release_id=parent_release_id,
        document_count=chunk_count,
        chunk_count=chunk_count,
        embedding_model="",
        embedding_dim=0,
        chunk_strategy="canonical",
        chunk_strategy_version="v1",
        lancedb_table="",  # filled by the hook after tagging
    )


# ---------------------------------------------------------------------------
# Hook builder
# ---------------------------------------------------------------------------

def build_publish_hook(
    *,
    app: Any,
    releases_root: Path,
    releases_dir: Path,
    tracker: Any,
    default_corp: Path,
) -> Callable[[Path, Path], None]:
    """Return the ``publish`` callable wired into ``ReindexJob``.

    Parameters
    ----------
    app:
        The FastAPI app (or a test app that carries the same attributes).
        The hook reads ``app.state.request_tracker``,
        ``app.state.build_retriever`` and the ``app._swap_runtime_services``
        seam from it; all three are already present on a production app.
    releases_root:
        Canonical-store root (``<root>``); the releases live under
        ``releases_root / RELEASES_DIRNAME``.
    releases_dir:
        ``<releases_root> / RELEASES_DIRNAME`` — passed explicitly so the
        hook does not re-derive it and silently disagree with the runner.
    tracker:
        The ``RequestTracker`` the hook drains before swapping.  Usually
        ``app.state.request_tracker``; passed in so the hook can be built in
        tests without a live app.
    default_corp:
        The serving corpus directory (``release_status.DEFAULT_CORPUS_DIR``).
    """
    releases_dir = Path(releases_dir)
    releases_dir.mkdir(parents=True, exist_ok=True)

    # The release id the hook assigns: second-precision timestamp so two
    # publishes in the same second get distinct ids.
    _counter = 0

    def _next_release_id() -> str:
        nonlocal _counter
        ts = time.strftime("%Y%m%d%H%M%S", time.gmtime())
        seq = _counter
        _counter += 1
        return f"{ts}-{seq:04d}"

    # ------------------------------------------------------------------
    # The async drain-gate + swap body, scheduled onto the app's loop
    # ------------------------------------------------------------------
    async def _drain_and_swap(new_retriever: Any) -> None:
        """Run the gate + swap on the app's event loop.

        ``_reload_lock`` is the same ``asyncio.Lock`` the admin reload and
        rollback routes hold, so a publish can never overlap with a manual
        reload or a rollback.
        """
        from src.routers.admin import _reload_lock

        if tracker is None:
            # No request tracker on this app (unit-test / hand-mounted app):
            # drain is skipped, the swap still runs under the shared lock so
            # it cannot overlap a manual reload/rollback gate.
            async with _reload_lock:
                swap = getattr(app, "_swap_runtime_services", None)
                if swap is not None:
                    swap(new_retriever)
                _clear_query_cache(app)
            return

        async with _reload_lock:
            async with tracker.gate(timeout=_PUBLISH_DRAIN_TIMEOUT_S) as drained:
                if not drained:
                    logger.warning(
                        f"publish: drain timed out after "
                        f"{_PUBLISH_DRAIN_TIMEOUT_S}s "
                        f"(in_flight={tracker.in_flight}); "
                        "proceeding — the publish path has no force flag, "
                        "the old corpus keeps serving until this completes"
                    )
                # Swap the runtime: rebind every query-path alias to the new
                # retriever.  This is main.py's one-call swap seam.
                swap = getattr(app, "_swap_runtime_services", None)
                if swap is not None:
                    swap(new_retriever)
                # Clear the query cache: answers cached under the old release
                # must not be served after the swap.
                _clear_query_cache(app)
    # The synchronous hook body
    # ------------------------------------------------------------------
    def publish_hook(staged: Path, corpus_dir: Path) -> None:
        """ReindexJob._hand_over calls this with (staged_dir, corpus_dir).

        Any exception here propagates into ``ReindexJob.run``'s
        ``except BaseException`` block, which records ``job.state = FAILED``
        and re-raises.  The runner then keeps the outbox entry so the next
        tick retries — exactly the ticket's failure contract.
        """
        from scripts.ingest_lancedb import ingest_canonical_corpus

        staged = Path(staged)
        corpus_dir = Path(corpus_dir)

        # ── 1. Copy staged corpus → serving corpus directory ───────────────
        # The default _publish does this file-by-file with .tmp siblings for
        # atomicity.  We do the same here so the serving corpus is updated
        # before the ingest step reads it.
        corpus_dir.mkdir(parents=True, exist_ok=True)
        from src.services.reindex_job import (
            CHUNKS_FILENAME,
            CHUNK_MANIFEST_FILENAME,
            GRAPH_EDGES_FILENAME,
            VECTORS_FILENAME,
        )
        order = (
            CHUNKS_FILENAME,
            VECTORS_FILENAME,
            GRAPH_EDGES_FILENAME,
            CHUNK_MANIFEST_FILENAME,
        )
        staged_files: list[tuple[Path, Path]] = []
        for name in order:
            src = staged / name
            dst = corpus_dir / name
            tmp = dst.with_name(dst.name + ".tmp")
            if src.is_file():
                tmp.write_bytes(src.read_bytes())
                staged_files.append((tmp, dst))
        for tmp, dst in staged_files:
            os.replace(tmp, dst)

        # ── 2. Ingest into LanceDB ──────────────────────────────────────────
        # Raises IngestCorpusError if the corpus is stale or empty.
        ingest_canonical_corpus(corpus_dir)

        # ── 3. Build the release artefacts ──────────────────────────────────
        release_id = _next_release_id()

        # Read the parent from the current pointer, if any.
        parent_release_id: Optional[str] = None
        pointer_path = releases_dir / POINTER_FILENAME
        if pointer_path.is_file():
            try:
                parent_release_id = json.loads(
                    pointer_path.read_text(encoding="utf-8")
                ).get("release_id")
            except (json.JSONDecodeError, OSError):
                logger.warning(
                    f"publish: could not read pointer {pointer_path}; "
                    "setting parent_release_id=None"
                )

        # Tag the LanceDB table at its current version so a rollback can
        # restore it.  The table was just ingested above, so the current
        # version already contains the new rows.
        lancedb_version: int = 0
        try:
            from config.settings import RAGSettings

            import lancedb as _lancedb

            settings = RAGSettings()
            db = _lancedb.connect(settings.LANCEDB_URI)
            target = os.environ.get("LANCEDB_UNIFIED_TABLE", "husc_chunks")
            if target in _lancedb_table_names(db):
                table = db.open_table(target)
                lancedb_version = tag_release(table, release_id)
            else:
                logger.info(
                    f"publish: LanceDB table {target!r} not found; "
                    "skipping tag (no table to tag)"
                )
        except Exception as exc:
            # A tagging failure must not block the file-level publish: the
            # corpus is on disk and the ingest already merged the rows.
            # The manifest will record lancedb_version=0 and rollback will
            # have to refuse (TargetNotTaggedError) until the operator fixes
            # the table manually.
            logger.warning(f"publish: failed to tag LanceDB release: {exc}")

        # Write the manifest + pointer.
        manifest = make_release_manifest(
            release_id=release_id,
            corpus_dir=corpus_dir,
            parent_release_id=parent_release_id,
        )
        manifest = _replace_with_version(manifest, lancedb_table_name(), lancedb_version)
        release_dir = releases_dir / release_id
        write_manifest(release_dir, manifest)
        publish_pointer(pointer_path, release_id)
        logger.info(
            f"publish: release {release_id} created "
            f"(parent={parent_release_id or 'none'}, "
            f"lancedb_version={lancedb_version})"
        )

        # ── 4. Drain-gate + runtime swap, under the shared lock ────────────
        # Build the new retriever over the freshly tagged table.  ``build_retriever``
        # is async and schedules LanceDB work in a thread; the gate + swap are
        # one coroutine that runs under the admin route's ``_reload_lock`` and
        # inside ``tracker.gate`` so no query can read the old runtime mid-swap.
        builder: Any = getattr(app.state, "build_retriever", None)
        if builder is not None:
            new_retriever = _build_retriever_sync(app, builder)
            _run_on_app_loop(app, _drain_and_swap(new_retriever))
        else:
            # No retriever builder wired (unit tests, minimal app): skip the
            # swap.  The release artefacts are already on disk; the serving
            # path keeps the old retriever until a restart or manual reload.
            logger.info(
                "publish: no build_retriever on app.state; "
                "release artefacts written, runtime not swapped"
            )

        # ── 5. T07: resolve open unanswered-queue items the new corpus answers ──
        #
        # Invariant: this step runs ONLY after ingest_canonical_corpus has
        # returned successfully, the release artefacts are on disk, and the
        # runtime swap is complete (or explicitly skipped) — i.e. the new
        # corpus is genuinely serving. A publish whose ingest or swap fails
        # raises earlier in this function and never reaches here, so a
        # failed publish cannot falsely close a queue item.
        #
        # Best-effort: a resolve failure (missing store, missing queue, IO
        # error) logs a warning and is swallowed — the corpus is already
        # serving; the resolution is a side effect, never a gate.
        _resolve_unanswered_items(app, corpus_dir)

        # ── 6. T16: append the release-bound serve-probe entry (K3) ─────────
        # Same contract as step 5: best-effort side effect, never a gate.
        # The probe joins upload time (registry) to serve time (query path);
        # the query side stamps ``served_at`` when a chunk of this release
        # is returned (see src/services/serve_probe.py).
        _append_probe_entry(
            app, releases_root=Path(releases_root),
            release_id=release_id, manifest=manifest, corpus_dir=corpus_dir)
    return publish_hook



def _append_probe_entry(
    app: Any,
    *,
    releases_root: Path,
    release_id: str,
    manifest: Any,
    corpus_dir: Path,
) -> None:
    """T16: append ONE probe-log entry for a freshly published release (K3).

    Called from ``publish_hook`` at step 6, after the runtime swap — the
    release is genuinely serving, so the entry's ``served_at: null`` is a
    promise the query path fulfils, not a guess. Best-effort and swallowed:
    a probe failure must never mark the publish FAILED.

    Probe derivation: record = first ``doc_id`` in the just-published
    ``chunks.jsonl``; chunk_release_id = that chunk's own
    ``metadata.release_id`` (the build-time stamp queries actually see —
    the hook's publish-time id differs, so the query side matches
    either); probe_question = first 200 chars of that record's
    text; upload_received_at = the source-document registry's
    ``registered_at`` for the record's ``source_document_id`` (``null`` +
    ``upload_received_at_known=false`` when the link is missing — never
    invented); published_at = the manifest clock.
    """
    try:
        from src.services.serve_probe import append_probe_entry
    except ImportError as exc:
        logger.warning(f"publish: T16 probe skipped (import failed): {exc}")
        return
    try:
        record_id: Optional[str] = None
        chunk_release_id: Optional[str] = None
        chunks_path = corpus_dir / "chunks.jsonl"
        if chunks_path.is_file():
            with open(chunks_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        first = json.loads(line)
                    except (json.JSONDecodeError, AttributeError):
                        continue
                    if not isinstance(first, dict):
                        continue
                    # The chunk's own build-time release stamp (the id queries
                    # see in metadata.release_id) — the query side matches
                    # this OR the hook's publish-time id, so a release
                    # re-published from older-built chunks still closes.
                    meta = first.get("metadata")
                    if chunk_release_id is None and isinstance(meta, dict) and meta.get("release_id"):
                        chunk_release_id = str(meta["release_id"])
                    doc_id = first.get("doc_id")
                    if doc_id:
                        record_id = str(doc_id)
                        break
        text_snippet: Optional[str] = None
        source_document_id: Optional[str] = None
        store = getattr(app.state, "canonical_record_store", None)
        if store is not None and record_id is not None:
            try:
                record = store.get_record(record_id).latest
            except Exception:
                record = None
            if record is not None:
                text = getattr(record, "text", None)
                if isinstance(text, str) and text.strip():
                    text_snippet = text.strip()[:200]
                source_document_id = getattr(record, "source_document_id", None)
        upload_received_at: Optional[str] = None
        upload_known = False
        registry = getattr(app.state, "source_document_registry", None)
        if registry is not None and source_document_id:
            try:
                entry = registry.get(str(source_document_id))
            except Exception:
                entry = None
            if entry is not None and getattr(entry, "registered_at", None) is not None:
                upload_received_at = entry.registered_at.isoformat()
                upload_known = True
        published_at = getattr(manifest, "created_at", None)
        published_iso = (
            published_at.isoformat()
            if hasattr(published_at, "isoformat")
            else (str(published_at) if published_at else None)
        )
        append_probe_entry(
            data_dir=releases_root,
            release_id=release_id,
            chunk_release_id=chunk_release_id,
            record_id=record_id,
            probe_question=text_snippet,
            upload_received_at=upload_received_at,
            upload_received_at_known=upload_known,
            published_at=published_iso,
        )
        logger.info(
            f"publish: T16 probe entry for release {release_id} "
            f"(record={record_id or 'none'}, upload_known={upload_known})"
        )
    except Exception as exc:
        logger.warning(f"publish: T16 probe failed (corpus already serving): {exc}")

def _resolve_unanswered_items(app: Any, corpus_dir: Path) -> None:
    """T07: resolve open queue items that the newly served corpus answers.

    Called from ``publish_hook`` at step 5, after the corpus is genuinely
    serving. For each served record_id that carries a ``values.source_trace_id``
    in its structured payload, any open queue item under that trace is marked
    resolved with this record's id as the resolving record.

    Best-effort: any exception (missing store, missing queue service, IO
    error, ``ItemNotFoundError``) is logged and swallowed — a resolve failure
    must never mark the publish FAILED because the corpus is already live.
    """
    from src.services.unanswered_queue import ItemNotFoundError

    queue = getattr(app.state, "unanswered_queue", None)
    if queue is None:
        logger.info("publish: T07 resolve skipped — no unanswered_queue on app.state")
        return

    # Collect served record_ids from chunks.jsonl (tolerant of missing file).
    chunks_path = corpus_dir / "chunks.jsonl"
    record_ids: set[str] = set()
    if chunks_path.is_file():
        try:
            with open(chunks_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc_id = json.loads(line).get("doc_id")
                        if doc_id:
                            record_ids.add(doc_id)
                    except (json.JSONDecodeError, AttributeError):
                        continue
        except OSError:
            logger.warning("publish: T07 resolve — could not read chunks.jsonl, skipping")
            return

    store = getattr(app.state, "canonical_record_store", None)
    if store is None:
        logger.info("publish: T07 resolve skipped — no canonical_record_store on app.state")
        return

    # Build a trace_id → record_id map for records that carry source_trace_id.
    trace_to_record: dict[str, str] = {}
    for rid in record_ids:
        try:
            record = store.get_record(rid).latest
        except Exception:
            continue
        # CanonicalRecord has extra="allow"; the structured payload (including
        # source_trace_id) is stored in the record's "values" dict.
        values = getattr(record, "values", None)
        if not isinstance(values, dict):
            values = getattr(record, "model_extra", None)
        if not isinstance(values, dict):
            continue
        trace_id = values.get("source_trace_id")
        if isinstance(trace_id, str) and trace_id.strip():
            trace_to_record[trace_id] = rid

    if not trace_to_record:
        return

    # Iterate open items once; resolve only those whose trace_id is in the map.
    try:
        for item in queue.unanswered():
            rid = trace_to_record.get(item.trace_id)
            if rid is None:
                continue
            queue.resolve(item.trace_id, rid)
            logger.info(
                f"publish: T07 resolved trace {item.trace_id!r} "
                f"with record {rid!r}"
            )
    except ItemNotFoundError as exc:
        # Item was resolved concurrently (idempotent-safe); log and move on.
        logger.info(f"publish: T07 resolve — {exc}; already resolved, skipping")
    except Exception as exc:
        logger.warning(f"publish: T07 resolve failed (corpus already serving): {exc}")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _lancedb_table_names(db) -> set[str]:
    listed = db.list_tables()
    names = listed.tables if hasattr(listed, "tables") else listed
    return set(names)


def lancedb_table_name() -> str:
    from config.settings import RAGSettings

    return RAGSettings().LANCEDB_TABLE


def _replace_with_version(
    manifest: ReleaseManifest, table: str, version: int
) -> ReleaseManifest:
    """Patch the manifest with the concrete table name and version."""
    from dataclasses import replace

    return replace(manifest, lancedb_table=table, lancedb_tag=f"release-{manifest.release_id}", lancedb_version=version)


def _build_retriever_sync(app: Any, builder: Any) -> Any:
    """Run the async ``builder(table_name)`` and return the retriever.

    Uses ``asyncio.run`` in a fresh loop when no app loop is registered —
    the TestClient case.  In production the uvicorn loop is captured at
    startup and scheduled onto via ``run_coroutine_threadsafe``.
    """
    table = lancedb_table_name()
    loop: Optional[asyncio.AbstractEventLoop] = getattr(app.state, "_publish_event_loop", None)
    if loop is not None and loop.is_running():
        fut = asyncio.run_coroutine_threadsafe(builder(table), loop)
        return fut.result(timeout=30)
    # No running loop (TestClient, unit test): run in a fresh one.
    return asyncio.run(builder(table))


def _run_on_app_loop(app: Any, coro) -> None:
    """Schedule ``coro`` onto the app's event loop; block until it finishes."""
    loop: Optional[asyncio.AbstractEventLoop] = getattr(app.state, "_publish_event_loop", None)
    if loop is not None and loop.is_running():
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        fut.result(timeout=60)
    else:
        asyncio.run(coro)


def _clear_query_cache(app: Any) -> None:
    """Clear the module-level ``query_cache`` in ``src.main`` if reachable."""
    try:
        import main as _main

        if _main.query_cache is not None:
            _main.query_cache.clear()
            logger.info("publish: query cache cleared")
    except (ImportError, AttributeError):
        # main.py not importable (unit-test context): try app.state fallback.
        cache: Optional[QueryCache] = getattr(app.state, "query_cache", None)
        if cache is not None:
            cache.clear()


# ---------------------------------------------------------------------------
# Registration helper — called from lifecycle.py after the app is ready
# ---------------------------------------------------------------------------

def register_publish_event_loop(app: Any, loop: asyncio.AbstractEventLoop) -> None:
    """Store the running event loop so the hook can schedule async work on it.

    Called from ``lifespan`` after ``configure_canonical_record_store`` runs,
    while the uvicorn loop is live.
    """
    app.state._publish_event_loop = loop
