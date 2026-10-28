"""App lifecycle: lazy service registry + startup/shutdown hooks.
@spec(S10.B2)
"""
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

# Ticket 12 wiring lives in this module, and under the launch convention
# uvicorn actually uses (cwd=rag2025, sys.path=['src','.']) the package-root
# form below does not resolve: there is no `rag2025` on the path, so
# `rag2025.src.services.embedding` raises ModuleNotFoundError and main.py's
# guarded import of *this* module fails, booting the app with no
# `canonical_record_store` — every store-dependent admin route 503s. The
# package-root form is tried first because the test suite imports this module
# that way and asserts class identity against it; the src-relative form is the
# production fallback. Same dual-convention shape as the in-function imports
# below and src/routers/admin.py:84-94.
try:
    from rag2025.config.settings import RAGSettings
    from rag2025.src.services.embedding import EmbeddingService
    from rag2025.src.services.llm_client import UnifiedLLMClient, get_llm_client
    from rag2025.src.services.admin_rollback import RetrievalUnavailableError
except ModuleNotFoundError:
    from config.settings import RAGSettings  # type: ignore
    from services.embedding import EmbeddingService  # type: ignore
    from services.llm_client import (  # type: ignore
        UnifiedLLMClient,
        get_llm_client,
    )
    from services.admin_rollback import RetrievalUnavailableError  # type: ignore

# T07: repo root the source-document registry resolves document ids
# against. Same expression as ``src/routers/admin_source_documents.py``
# (``parents[3]`` of this file) — the registry only uses it to record
# where each registered document's file lives, so the two must agree.
REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class AppServices:
    """Lazy service container — services constructed on first access.
    @spec(S10.B2)
    """
    _settings: Optional[RAGSettings] = None
    embedding: Optional[EmbeddingService] = None
    llm_client: Optional[UnifiedLLMClient] = None

    def get_settings(self) -> RAGSettings:
        if self._settings is None:
            self._settings = RAGSettings()
        return self._settings

    def get_embedding(self) -> EmbeddingService:
        if self.embedding is None:
            self.embedding = EmbeddingService(self.get_settings())
        return self.embedding

    def get_llm_client(self, force_model: Optional[str] = None) -> UnifiedLLMClient:
        if self.llm_client is None:
            self.llm_client = get_llm_client(force_model=force_model)
        return self.llm_client


async def startup(services: AppServices) -> None:
    """Fast startup — services constructed lazily, no model load yet.
    @spec(S10.B2)
    """
    pass  # lazy: nothing to do


async def shutdown(services: AppServices) -> None:
    """Cleanup on shutdown."""
    # UnifiedLLMClient uses per-call AsyncOpenAI instances (no persistent pool to close).
    # If a close() method is added later, call it here.
    pass


def configure_canonical_record_store(app, settings: Optional[RAGSettings] = None):
    """Attach the versioned canonical admissions store to ``app.state``.

    Ticket 12: the admin lifecycle endpoints read their store from app state
    (same injection pattern as reload-table's ``state.request_tracker``), so
    something has to construct it at startup. The store is file-backed
    (``<root>/records/rec-<encoded-id>/vNNNN.json``), so no database is needed
    and spec Further-Note #10 — which DB the canonical store runs on — does not
    block this.

    ``CANONICAL_STORE_ROOT`` is the configured root; the directory is created
    if missing. A failure here is raised, not swallowed: a store that silently
    failed to configure would turn every lifecycle write into a 503 at request
    time, which is exactly the failure mode this wiring exists to remove. The
    caller (app startup) decides how loud to be.

    Returns the constructed store so a test can assert against the same object
    the app sees.
    """
    resolved = settings or RAGSettings()
    root = Path(resolved.CANONICAL_STORE_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    # Same dual-convention fallback the admin router already uses for
    # AppServices: the project layout differs between "src on sys.path" and
    # "repo root on sys.path", and this function must work in both.
    try:
        from src.config.admin_identity import four_eyes_enabled
        from src.infrastructure.audit_ledger import AuditLedger
        from src.infrastructure.canonical_record_store import CanonicalRecordStore
        from src.services.admin_lifecycle import AdminLifecycleService
        from src.services.admin_rollback import (
            RELEASES_DIRNAME,
            AdminRollbackService,
            LanceDBRetrievalRollback,
            ReleaseGraphRollback,
            RetrievalRollbackError,
            RetrievalUnavailableError,
        )
        from src.services.abstention_policy import configure_conflict_source
        from src.services.conflict_queue import ConflictQueueService
        from src.services.idempotency import IdempotencyStore
        from src.services.reindex_outbox import OUTBOX_FILENAME, ReindexOutbox
        from src.services.reindex_runner import ReindexRunner
        from src.services.publish_hook import build_publish_hook
        from src.services.release_status import ReleaseStatusService
        from src.infrastructure.source_document_registry import (  # type: ignore
            SourceDocumentRegistry,
        )
    except ModuleNotFoundError:
        from rag2025.src.config.admin_identity import (  # type: ignore
            four_eyes_enabled,
        )
        from rag2025.src.infrastructure.audit_ledger import (  # type: ignore
            AuditLedger,
        )
        from rag2025.src.infrastructure.canonical_record_store import (  # type: ignore
            CanonicalRecordStore,
        )
        from rag2025.src.services.admin_lifecycle import (  # type: ignore
            AdminLifecycleService,
        )
        from rag2025.src.services.admin_rollback import (  # type: ignore
            RELEASES_DIRNAME,
            AdminRollbackService,
            LanceDBRetrievalRollback,
            ReleaseGraphRollback,
            RetrievalRollbackError,
            RetrievalUnavailableError,
        )
        from rag2025.src.services.abstention_policy import (  # type: ignore
            configure_conflict_source,
        )
        from rag2025.src.services.conflict_queue import (  # type: ignore
            ConflictQueueService,
        )
        from rag2025.src.services.idempotency import (  # type: ignore
            IdempotencyStore,
        )
        from rag2025.src.services.reindex_outbox import (  # type: ignore
            OUTBOX_FILENAME,
            ReindexOutbox,
        )
        from rag2025.src.services.reindex_runner import (  # type: ignore
            ReindexRunner,
        )
        from rag2025.src.services.publish_hook import (  # type: ignore
            build_publish_hook,
        )
        from rag2025.src.services.release_status import (  # type: ignore
            ReleaseStatusService,
        )

        from rag2025.src.infrastructure.source_document_registry import (  # type: ignore
            SourceDocumentRegistry,
        )
    # T07 (spec 02 §1): the source-document registry, under the same
    # root as the ledger and the outbox (``<root>/documents/``), built
    # here so a store copy never arrives without it. The lifecycle
    # service needs it to generate a real source document for HOI_DAP
    # creates; the source-documents router builds its own lazily, so
    # this one is handed to the lifecycle service only.
    registry = SourceDocumentRegistry(root, REPO_ROOT)
    app.state.source_document_registry = registry

    store = CanonicalRecordStore(root)
    app.state.canonical_record_store = store

    # Ticket 14: the append-only audit ledger. It lives beside the records
    # under the same root, so the two are published and rolled back together
    # and a store copy never arrives without its history. Handed to the
    # lifecycle service so transitions append events as they happen, and put
    # on app.state so the read-only audit router can serve them; that router
    # answers 503 when this is absent, so an unwired app refuses rather than
    # reporting an empty history.
    ledger = AuditLedger(root / "audit" / "ledger.jsonl")
    app.state.audit_ledger = ledger

    # Ticket 15: the re-index outbox. Constructed here, at startup, rather than
    # lazily on first use — the one thing that must not happen is a publish
    # landing while the outbox path is still unresolved, because that
    # publication's affected ids would never be queued and nothing would say
    # so. Built beside the ledger, under the same root, for the same reason the
    # ledger is: they travel with the records. Put on app.state for the
    # trigger endpoint (ticket 19) and the re-index job to read.
    outbox = ReindexOutbox(root / "outbox" / OUTBOX_FILENAME)
    app.state.reindex_outbox = outbox

    # Ticket 15 criterion 7: something in production RUNS the job. The outbox
    # above only records what changed; without a runner the corpus is rebuilt
    # when a human remembers to invoke a script, which is the gap this closes.
    #
    # Built here because this is the established startup seam for everything
    # that must exist before the first request. The embedder is a *provider*,
    # not a value: main.py assigns ``app.state.embedding_service`` after this
    # function returns, so a capture at startup would be a permanent ``None``.
    # Resolved per tick, the worker picks the model up the moment it exists and
    # abstains (logs, sleeps) until then — which is also what makes the ~10
    # router fixtures that call this function directly spawn a thread that never
    # touches the real corpus.
    #
    # ``job_sink`` mirrors each fresh job onto ``app.state.reindex_job``, the
    # name src/routers/admin.py already reads with getattr, so the index-jobs
    # screen reports pending/running/failed/active with no router change.
    # A repeated startup call must not stack workers: idempotency is per
    # runner object, and each call builds a new one, so the previous runner is
    # stopped before the new thread starts. In production this covers any
    # second configure (reload, re-mount); under pytest it bounds the damage
    # if the autorun gate is ever enabled around this seam.
    previous = getattr(app.state, "reindex_runner", None)
    if previous is not None:
        previous.stop()
    releases_root = root / RELEASES_DIRNAME
    publish = build_publish_hook(
        app=app,
        releases_root=root,
        releases_dir=releases_root,
        tracker=getattr(app.state, "request_tracker", None),
        default_corp=root / "canonical_corpus",
    )
    runner = ReindexRunner(
        store=store,
        outbox=outbox,
        embedder_provider=lambda: getattr(app.state, "embedding_service", None),
        releases_root=releases_root,
        job_sink=lambda job: setattr(app.state, "reindex_job", job),
        publish=publish,
    )
    app.state.reindex_runner = runner
    runner.start()

    # One service instance per app so the emitted-event stream accumulates
    # across requests. The four-eyes switch is passed as the config callable
    # (ticket 36): the service consults it at each check, so the control
    # reflects the deployment's current environment rather than a snapshot
    # taken before the app served its first request.
    app.state.admin_lifecycle_service = AdminLifecycleService(
        store, four_eyes=four_eyes_enabled, ledger=ledger, outbox=outbox,
        registry=registry,
    )

    # Ticket 22: the two-level rollback. Built here for the same reason the
    # outbox is — the releases root has to be resolved before anything can
    # publish or roll back, and a service that resolved it lazily could move the
    # pointer under one root while the publish path wrote another.
    #
    # The releases live under the canonical store root, beside the ledger and
    # the outbox: a copy of the store then carries the records, their history and
    # their releases together, which is the same argument ticket 14 made for the
    # ledger.
    #
    # The retrieval half is deliberately late-bound. main.py owns both the
    # serving retriever and the swap seam, and it builds the retriever *after*
    # warmup — so the service gets callables that resolve them when a rollback
    # runs, never objects captured at startup (a captured `None` would make every
    # rollback fail with a confusing "no serving retriever"). The accessors are
    # main.py's `_health_lancedb` / `_swap_runtime_services`; an app that does not
    # expose them (a hand-mounted test app) fails loudly at call time rather than
    # swapping something else.
    app.state.admin_rollback_service = AdminRollbackService(
        releases_root=root / RELEASES_DIRNAME,
        retrieval=LanceDBRetrievalRollback(
            retriever_provider=partial(_serving_retriever, app),
            swap=partial(_swap_via_app, app),
        ),
        graph=ReleaseGraphRollback(
            pipeline_provider=partial(_serving_pipeline, app),
        ),
        ledger=ledger,
    )

    # Ticket 27: request-level idempotency for the mutating admin routes. Built
    # here, under the same root as the ledger and the outbox, so a copy of the
    # store carries the records, their history, their releases and the keys that
    # de-duplicate the requests that changed them.
    #
    # The store is attached to app.state rather than passed to the router: the
    # routes read it per request through the guard, and an app that never
    # configures it refuses a keyed request with 503 rather than silently
    # running the operation twice.
    app.state.idempotency_store = IdempotencyStore(root)

    # Ticket 11: the abstention policy's conflict feed. Rule 3 refuses to
    # self-select when two official sources contradict each other, but the
    # policy module is pure and cannot import the queue itself — it reads a
    # zero-arg callable registered here. Without this call the default is an
    # empty feed, which means criterion 3 would be correct in tests and inert in
    # production: the policy would never see a real conflict and would answer
    # straight past it. Registered at startup for the same reason the outbox is
    # — a lazily-resolved feed could miss the window where the first conflict is
    # already queued and being served.
    #
    # `items()` is deliberately a read of the persisted queue, not a
    # detection pass: generation must not mutate the queue, and `refresh()`
    # writes. A queue file that is absent yields `[]`, so a fresh deployment
    # abstains on nothing rather than failing.
    from services.abstention_policy import configure_conflict_source

    conflict_queue_service = ConflictQueueService(store, ledger)
    app.state.conflict_queue_service = conflict_queue_service
    configure_conflict_source(conflict_queue_service.items)

    # Ticket T10: the refusal trailer's "last updated" date. The abstention
    # policy renders the footer but is pure and cannot read the release pointer
    # itself — it calls a zero-arg provider registered here, exactly like the
    # conflict feed above. Without this the hotline shows but the serving
    # release's date never does, so the operator-facing promise ("the date the
    # answer is based on") would be correct in tests and inert in production.
    #
    # The provider reads the pointer *per call*, not once at startup: the
    # active release changes on every publish, and a captured value would pin
    # the footer to the release that happened to be live at boot. A missing or
    # unreadable pointer yields None — the footer then omits the date line
    # rather than failing the refusal it is decorating.
    #
    # The policy file is reachable under two module identities in this codebase
    # (``services.abstention_policy`` when ``src`` is on ``sys.path``, and
    # ``src.services.abstention_policy`` from main.py's absolute import). They
    # are the same file but distinct module objects, and the footer is rendered
    # from both — the generator abstains through ``services.*`` while main.py
    # appends the trailer through ``src.*``. Registering the provider on only
    # one would leave the other's date line blank, so every importable identity
    # gets it (the T10 test does the same for the same reason).
    from services.abstention_policy import configure_release_date_source

    def _serving_release_date() -> Optional[str]:
        try:
            status = ReleaseStatusService(releases_root=root / RELEASES_DIRNAME).status()
        except Exception:
            return None
        if status.active is None:
            return None
        return status.active.created_at

    configure_release_date_source(_serving_release_date)
    try:
        from src.services.abstention_policy import (
            configure_release_date_source as _configure_src,
        )
    except ModuleNotFoundError:
        pass
    else:
        _configure_src(_serving_release_date)

    return store


def _runtime_accessor(app, name: str):
    """main.py's lazy accessor for a runtime global, or ``None``.

    The accessors are set on the app object when ``src/main.py`` is imported
    (``app._health_lancedb``), so a module can read the module globals without
    importing main.py by name — uvicorn runs it as ``main`` while tests load it
    under their own names.
    """
    return getattr(app, name, None)


def _serving_retriever(app):
    """The retriever the process is serving from, via main.py's accessor.

    Resolved per call, never captured: main.py builds the retriever during
    warmup, after this wiring runs.
    """
    accessor = _runtime_accessor(app, "_health_lancedb")
    if accessor is None:
        raise RetrievalUnavailableError(
            "the app exposes no _health_lancedb accessor, so the serving "
            "retriever cannot be resolved"
        )
    return accessor()


def _serving_pipeline(app):
    """The pipeline the process is serving from, via main.py's accessor.

    Resolved per call for the same reason the retriever is: main.py builds the
    pipeline during warmup, after this wiring runs. Unlike the retriever, an
    unset accessor is **not** an error — a deployment whose app exposes no
    pipeline serves no graph, and the rollback's graph level reports
    ``"not-serving"`` rather than failing the whole rollback.
    """
    accessor = _runtime_accessor(app, "_health_pipeline")
    return accessor() if accessor is not None else None


def _swap_via_app(app, retriever):
    """Rebind the runtime through main.py's swap seam.

    Raises rather than degrading: a fallback that assigned something else would
    report a successful rollback while every query kept the old retriever, which
    is the P1 bug ticket 22 Part 0 exists to remove.
    """
    seam = _runtime_accessor(app, "_swap_runtime_services")
    if seam is None:
        raise RetrievalRollbackError(
            "the app exposes no _swap_runtime_services seam, so a rollback cannot "
            "rebind the retriever every query reads"
        )
    return seam(retriever)
