"""ReindexRunner - ticket 15 criterion 7: something in production RUNS the job.

Before this module, the outbox drained only when a human remembered to run a
script: ``ReindexJob`` existed, ``ReindexOutbox`` recorded every publication,
the admin screen reported the queue - and nothing connected them. This is that
missing connection: a plan-driven tick, plus a background worker that performs
it without being told.

The design is deliberately thin. All of the hard decisions already live in
``ReindexJob``:

* **what** to rebuild - :func:`src.services.reindex_job.plan_reindex`, from the
  pending outbox entries and the corpus stamp;
* **whether a run is warranted at all** - the same two questions, asked here as
  :meth:`ReindexRunner.due` so a tick that has nothing to do costs one read of
  each, and so the CLI can say "nothing due" instead of building a job to
  discover it has no work.

The runner never re-implements the merge, the publish, or the cursor. It builds
a ``ReindexJob`` over the release the process is serving and calls ``run()``.

Wiring
------
``configure_canonical_record_store`` constructs the runner and calls
``start()``. ``start()`` is gated by :func:`autorun_enabled`, so the ~10 router
fixtures that call the seam directly never spawn a thread; ``run_once()`` itself
is ungated and does real work regardless, which is what makes the runner testable
and what the operator CLI (``scripts/run_reindex.py``) relies on.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from loguru import logger

from src.domain.audit import LEGACY_RELEASE_ID
from src.services import release_status
from src.services.published_materializer import PublishedMaterializer
from src.services.reindex_job import (
    CorpusStamp,
    ReindexJob,
    read_corpus_stamp,
    stamp_is_stale,
)
from src.services.reindex_outbox import ReindexOutbox

#: Env vars (groundwork item 3b):
#: ``REINDEX_AUTORUN``       - "1|true|yes|on" enables the worker,
#:                             "0|false|no|off" disables it. Unset means ON in a
#:                             real process and OFF when pytest is imported, so
#:                             the router fixtures that call the wiring seam
#:                             directly never spawn a thread. The e2e test sets
#:                             it explicitly.
#: ``REINDEX_TICK_SECONDS``  - sleep after a tick that published something.
#: ``REINDEX_RETRY_SECONDS`` - sleep after a tick that did nothing, was skipped,
#:                             or failed; also the delay before the next attempt
#:                             while the embedder is not resolvable yet (the
#:                             wiring runs before ``app.state.embedding_service``
#:                             is assigned, so the first ticks legitimately
#:                             abstain and must pick the model up quickly).
AUTORUN_ENV = "REINDEX_AUTORUN"
TICK_SECONDS_ENV = "REINDEX_TICK_SECONDS"
RETRY_SECONDS_ENV = "REINDEX_RETRY_SECONDS"

DEFAULT_TICK_SECONDS = 300.0
DEFAULT_RETRY_SECONDS = 5.0
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def autorun_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether the background worker may run in this process.

    An explicit value always wins. With nothing set, the default is ON for a
    real process and OFF under pytest: the wiring seam is called directly by
    roughly ten router fixtures, and a thread spawned by each of them would
    keep polling a ``tmp_path`` that the test has already torn down. That is a
    recorded deviation from the ticket's "conftest sets 0" plan, which assumed
    the start call lived in ``main.lifespan``; ``main.py`` is contention-locked,
    so the gate has to live here instead.
    """
    source = os.environ if env is None else env
    raw = source.get(AUTORUN_ENV, "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    import sys

    return "pytest" not in sys.modules


@dataclass(frozen=True)
class RunnerOutcome:
    """What one tick did - returned, never raised for ordinary failures.

    ``due`` says the plan called for work, independent of whether the tick could
    perform it. ``action`` is the one field a caller branches on:

    ``"noop"``
        nothing was due (``due=False``);
    ``"ran"``
        a job ran; ``result`` says whether it is serving;
    ``"skipped"``
        work was due and could not be started (no embedder, no published
        records); ``reason`` names the gate;
    ``"failed"``
        a job ran and raised; ``error`` carries it and ``result`` may still be
        the partial record the job kept.

    ``retry_soon`` is the worker's own sleep policy and is never read by the CLI.
    """

    due: bool = False
    action: str = "noop"
    reason: str = ""
    result: Any | None = None
    error: str | None = None
    retry_soon: bool = False

    @property
    def ran(self) -> bool:
        return self.action == "ran"


class ReindexRunner:
    """Drains the outbox and rebuilds the corpus when the plan says so.

    ``store``
        The canonical record store the app is serving. Records come from it via
        ``PublishedMaterializer`` unless ``records`` names a JSONL directly.
    ``outbox``
        The publication queue. ``None`` removes the record-level trigger and
        leaves the stamp trigger working, which is the honest degraded shape
        rather than a crash.
    ``embedder_provider``
        ``() -> Any``, resolved per tick, never captured at construction: the
        wiring seam runs before ``app.state.embedding_service`` is assigned and
        the worker must pick the model up the moment it exists. Returns the
        *service object*; ``encode_documents`` is resolved off it (a bare
        callable is accepted too). ``None`` makes the tick skipped, never a crash.
    ``corpus_dir``
        Where the serving corpus lives. ``None`` resolves
        ``release_status.DEFAULT_CORPUS_DIR`` lazily at tick time - the same
        module global ``IndexJobsService`` binds, so the runner and the admin
        screen cannot disagree about which corpus is "the" corpus.
    ``releases_root``
        Where the release pointer lives; ``None`` means no pointer can be read
        and every run is recorded against ``LEGACY_RELEASE_ID``.
    ``job_sink``
        Called with the fresh ``ReindexJob`` before every ``run()`` - wired to
        ``app.state.reindex_job`` so ``getattr(state, "reindex_job", None)`` in
        the admin router lights up with no router change.
    ``records`` / ``materializer_factory`` / ``release_id_provider`` /
    ``work_dir``
        Test and CLI seams. Production passes none of them.
    """

    def __init__(
        self,
        *,
        store: Any = None,
        outbox: ReindexOutbox | None = None,
        embedder_provider: Callable[[], Any] | None = None,
        corpus_dir: str | Path | None = None,
        releases_root: str | Path | None = None,
        work_dir: str | Path | None = None,
        job_sink: Callable[[ReindexJob], None] | None = None,
        interval_seconds: float | None = None,
        retry_seconds: float | None = None,
        release_id_provider: Callable[[], str | None] | None = None,
        materializer_factory: Callable[[Any], Any] | None = None,
        records: str | Path | None = None,
        publish: Callable[[Path, Path], None] | None = None,
    ) -> None:
        if store is None and records is None and materializer_factory is None:
            raise ValueError(
                "ReindexRunner needs a store, a records path, or a materializer "
                "factory - a runner that cannot name its input would spin forever"
            )
        self._store = store
        self._outbox = outbox
        self._embedder_provider = embedder_provider
        self._corpus_dir = Path(corpus_dir) if corpus_dir is not None else None
        self._releases_root = (
            Path(releases_root) if releases_root is not None else None
        )
        self._work_dir = Path(work_dir) if work_dir is not None else None
        self._job_sink = job_sink
        self._release_id_provider = release_id_provider
        self._materializer_factory = materializer_factory
        self._records = Path(records) if records is not None else None
        self._publish = publish
        self._interval_seconds = (
            float(interval_seconds)
            if interval_seconds is not None
            else _env_float(TICK_SECONDS_ENV, DEFAULT_TICK_SECONDS)
        )
        self._retry_seconds = (
            float(retry_seconds)
            if retry_seconds is not None
            else _env_float(RETRY_SECONDS_ENV, DEFAULT_RETRY_SECONDS)
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.started = False
        self.disabled_reason: str | None = None

    # ── resolution ─────────────────────────────────────────────────────────

    def _resolve_corpus_dir(self) -> Path:
        if self._corpus_dir is not None:
            return self._corpus_dir
        # Read the attribute at tick time, not import time: the screen's corpus
        # dir is a module global that tests and operators rebind.
        return Path(release_status.DEFAULT_CORPUS_DIR)

    def _resolve_release_id(self) -> str:
        if self._release_id_provider is not None:
            release_id = self._release_id_provider()
        elif self._releases_root is not None:
            service = release_status.ReleaseStatusService(
                releases_root=self._releases_root
            )
            release_id = service.status().active_release_id
        else:
            release_id = None
        return release_id or LEGACY_RELEASE_ID

    def _resolve_embedder(self) -> Callable[[Sequence[str]], Sequence[Any]] | None:
        """The service's ``encode_documents``, or ``None``.

        The provider hands back the ``EmbeddingService`` object, not a bound
        method, because the object is what ``app.state`` holds. A bare callable
        is accepted so a test can inject one without a wrapper.
        """
        if self._embedder_provider is None:
            return None
        service = self._embedder_provider()
        if service is None:
            return None
        candidate = getattr(service, "encode_documents", service)
        return candidate if callable(candidate) else None

    def _materialize(self, release_id: str, output_dir: Path):
        factory = self._materializer_factory
        if factory is None:
            if self._store is None:
                return None
            factory = lambda _store: PublishedMaterializer(_store)  # noqa: E731
        return factory(self._store).materialize(release_id, output_dir)

    def _materialize_dir(self, release_id: str, corpus_dir: Path) -> Path:
        if self._work_dir is not None:
            return self._work_dir / release_id
        # A sibling of the corpus, never inside the staging root the read model
        # iterates - a materialize must not masquerade as a staged build.
        return corpus_dir.parent / f"{corpus_dir.name}.materialize" / release_id

    # ── the plan ───────────────────────────────────────────────────────────

    def due(self) -> tuple[bool, str]:
        """Whether a job run is warranted right now, and why in one phrase.

        The two triggers are the job's own: pending outbox entries (a
        record-level signal) or a corpus whose stamp no longer matches the code
        that would rebuild it (the only signal a parser/chunker change
        produces - a bump with an EMPTY queue still forces a full rebuild).
        """
        pending = len(self._outbox.pending()) if self._outbox is not None else 0
        corpus_dir = self._resolve_corpus_dir()
        built = read_corpus_stamp(corpus_dir)
        current = CorpusStamp.current()
        if stamp_is_stale(built, current):
            return True, (
                f"corpus stamp {'absent' if built is None else 'stale'}; "
                f"{pending} pending outbox entries"
            )
        if pending:
            return True, f"{pending} pending outbox entries"
        return False, "corpus current and outbox drained"

    # ── one tick ───────────────────────────────────────────────────────────

    def run_once(self) -> RunnerOutcome:
        """Evaluate the plan and, if due, run one ``ReindexJob`` to completion.

        Never raises for ordinary failures: a job that went ``FAILED`` is
        already recorded on the job object (and visible through the sink), and
        an infrastructure error comes back in ``outcome.error``. A corrupt
        outbox is deliberately allowed to propagate to a CLI operator - the
        worker loop catches it and keeps the service alive.
        """
        embedder = self._resolve_embedder()
        if embedder is None:
            return RunnerOutcome(
                due=True,
                action="skipped",
                reason="embedding service not available yet",
                retry_soon=True,
            )

        is_due, reason = self.due()
        if not is_due:
            return RunnerOutcome(due=False, action="noop", reason=reason)

        corpus_dir = self._resolve_corpus_dir()
        release_id = self._resolve_release_id()
        records = self._records
        if records is None:
            result = self._materialize(
                release_id, self._materialize_dir(release_id, corpus_dir)
            )
            if result is None:
                return RunnerOutcome(
                    due=True,
                    action="skipped",
                    reason="no store and no records path - nothing to index",
                    retry_soon=True,
                )
            if not result.is_materialized:
                # An empty release is not a reason to publish an empty corpus
                # over the serving one.
                return RunnerOutcome(
                    due=True,
                    action="skipped",
                    reason=f"release {release_id} has no published records "
                    f"({result.state.value})",
                    retry_soon=True,
                )
            records = result.records_path

        job = ReindexJob(
            records=records,
            corpus_dir=corpus_dir,
            release_id=release_id,
            embedder=embedder,
            outbox=self._outbox,
            publish=self._publish,
        )
        if self._job_sink is not None:
            self._job_sink(job)
        try:
            reindex_result = job.run()
        except Exception as exc:  # noqa: BLE001 - the worker must survive
            logger.error(f"reindex job raised: {exc}", exc_info=True)
            return RunnerOutcome(
                due=True,
                action="failed",
                reason=reason,
                error=f"{type(exc).__name__}: {exc}",
            )
        return RunnerOutcome(
            due=True, action="ran", reason=reason, result=reindex_result
        )

    # ── the worker ─────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Spawn the background worker. Idempotent; returns whether it runs.

        The first tick is immediate - self-activation has to be observable
        in-process, not after an arbitrary warm-up. When the gate is off no
        thread is spawned at all and ``disabled_reason`` says why.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            if not autorun_enabled():
                self.disabled_reason = f"{AUTORUN_ENV} not enabled for this process"
                self.started = False
                return False
            self.disabled_reason = None
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._loop, name="reindex-runner", daemon=True
            )
            self._thread.start()
            self.started = True
            return True

    def stop(self, timeout: float = 10.0) -> None:
        """Ask the worker to finish its current tick and join it."""
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def _loop(self) -> None:
        consecutive_errors = 0
        while not self._stop_event.is_set():
            try:
                outcome = self.run_once()
            except Exception as exc:  # noqa: BLE001 - a daemon must not die
                logger.error(f"reindex worker tick raised: {exc}", exc_info=True)
                consecutive_errors += 1
                self._stop_event.wait(self._backoff(consecutive_errors))
                continue
            if outcome.error is not None:
                consecutive_errors += 1
                self._stop_event.wait(self._backoff(consecutive_errors))
                continue
            consecutive_errors = 0
            if outcome.action == "ran":
                self._stop_event.wait(self._interval_seconds)
            else:
                # Both "noop" and "skipped" retry on the short interval: at
                # startup the embedder is not assigned yet, and the worker must
                # pick it up on the retry interval, not after a 300 s sleep.
                if outcome.action == "skipped":
                    logger.warning(f"reindex worker: {outcome.reason}")
                self._stop_event.wait(self._retry_seconds)

    def _backoff(self, consecutive_errors: int) -> float:
        return min(
            self._interval_seconds,
            self._retry_seconds * (2 ** (consecutive_errors - 1)),
        )
