"""TDD: /admin/rollback — two-level rollback (ticket 22) + the Part 0 swap seam.

Ticket 22's acceptance, asserted at the HTTP seam against the app ``src/main.py``
builds. That app, rather than a hand-mounted one, is deliberate: the retrieval
half of a rollback resolves the *serving retriever* and the *runtime swap seam*
from the real app (``app._health_lancedb`` / ``app._swap_runtime_services``), and
a hand-built app has neither — every rollback there fails with "no serving
retriever", which is itself the proof the wiring below is real.

What is covered:

- a rollback moves BOTH levels — ``releases/active_release.json`` and the LanceDB
  table the running retriever reads — in one request, and the response reports
  what happened per level;
- the table's *live data* is restored, read back through the same adapter the
  serving path uses, not merely the tag;
- the ROLLBACK audit event exists only after both levels have moved: a failed
  attempt leaves the ledger untouched, a successful one leaves exactly one event
  naming the authenticated actor and the reason;
- a failure in step D (retrieval) compensates step C (the pointer) and the 5xx
  names both halves — including the case where compensation itself fails, the
  only state in which the two levels can end up diverged;
- each failure maps onto its own status code (404 / 409 / 422 / 503), driven by
  real inputs wherever an input can reach the branch;
- a blank reason never reaches the store, and a role without
  ``records.rollback`` never reaches the handler;
- ``build_hybrid_service`` — the single construction path every swap uses —
  returns ``None`` when hybrid retrieval is off or its BM25 build fails, so a
  swap can never leave a sparse index over the table it just replaced.

Isolation: every test gets its own canonical-store root (``tmp_path``) and its
own LanceDB table, so the module is safe under ``-n`` and under any suite
ordering. No network and no LLM anywhere — the retrieval half is a real LanceDB
table, not a model.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import lancedb
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from rag2025.src.app.lifecycle import (  # noqa: E402
    configure_canonical_record_store,
)
from services.hybrid_search import HybridSearchService  # noqa: E402
from src.domain.audit import AuditAction, TargetKind  # noqa: E402
from src.domain.graph import GRAPH_SCHEMA_VERSION, KnowledgeGraph  # noqa: E402
from src.domain.release import ReleaseManifest  # noqa: E402
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_store import (  # noqa: E402
    publish_pointer,
    read_manifest,
    write_manifest,
)
from src.infrastructure.graph_artifact import publish_release_graph  # noqa: E402
from src.infrastructure.lancedb_versioning import tag_release  # noqa: E402
from src.services.admin_rollback import release_tag  # noqa: E402
from src.services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    LanceDBRetrieverConfig,
)

MAIN_PATH = ROOT / "src" / "main.py"
TABLE_NAME = "husc_chunks"

# One actor per role, so an assertion on ``actor_id`` is unambiguous. Only
# ``admin`` holds ``records.rollback`` in the matrix.
ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "publisher": ("cb.xuat-ban", "publisher"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"

# The lineage under test: R2 is active, R1 is the release behind it.
R1_ID = "R-2026-08-01"
R2_ID = "R-2026-09-01"
# A release that exists on disk but was never tagged in LanceDB.
R0_ID = "R-2026-07-01"

REASON = "khôi phục sau sự cố dữ liệu 2026"


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _configure_identities(monkeypatch) -> None:
    """Publish the actor registry the way a deployment does: environment."""
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": TOKENS[role], "actor_id": actor, "role": role}
                for role, (actor, _) in ACTORS.items()
            ]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)


def _row(row_id: str, text: str, release_id: str, *, metadata: dict | None = None) -> dict:
    """One LanceDB row in the shape the served table uses."""
    return {
        "id": row_id,
        "text": text,
        "source": "de-an-tuyen-sinh-2026.pdf",
        "chunk_id": row_id,
        "metadata_json": json.dumps(
            metadata if metadata is not None else {"release_id": release_id},
            ensure_ascii=False,
        ),
        "release_id": release_id,
        "vector": [0.1, 0.2, 0.3, 0.4],
    }


def _open_table(tmp_path: Path, rows: list[dict]):
    db = lancedb.connect(str(tmp_path / "lancedb"))
    return db.create_table(TABLE_NAME, data=rows)


def _retriever_over(tmp_path: Path) -> LanceDBRetriever:
    return LanceDBRetriever(
        LanceDBRetrieverConfig(
            uri=str(tmp_path / "lancedb"),
            table_name=TABLE_NAME,
            embedding_dim=4,
        )
    )


def _releases_dir(store_root: Path) -> Path:
    return store_root / "releases"


def _pointer_path(store_root: Path) -> Path:
    return _releases_dir(store_root) / "active_release.json"


def _active(store_root: Path) -> str:
    """The release the canonical pointer names, read from disk."""
    return json.loads(_pointer_path(store_root).read_text(encoding="utf-8"))["release_id"]


def _seed_release(
    store_root: Path,
    release_id: str,
    *,
    parent: str | None = None,
    lancedb_tag: str | None = None,
    manifest_id: str | None = None,
) -> None:
    """Write one release manifest under ``<root>/releases/<release_id>/``."""
    write_manifest(
        _releases_dir(store_root) / release_id,
        ReleaseManifest(
            release_id=manifest_id if manifest_id is not None else release_id,
            parent_release_id=parent,
            lancedb_tag=release_tag(release_id) if lancedb_tag is None else lancedb_tag,
        ),
    )


def _activate(store_root: Path, release_id: str) -> None:
    publish_pointer(_pointer_path(store_root), release_id)


def _ledger(store_root: Path) -> AuditLedger:
    """The same ledger file the app wiring writes — a second reader over it."""
    return AuditLedger(store_root / "audit" / "ledger.jsonl")


def _rollback_events(store_root: Path):
    return _ledger(store_root).find(action=AuditAction.ROLLBACK)


def _post_rollback(
    client: TestClient,
    *,
    reason: str = REASON,
    target: str | None = None,
    headers: dict[str, str] | None = None,
):
    body: dict = {"reason": reason}
    if target is not None:
        body["target_release_id"] = target
    return client.post(
        "/admin/rollback", json=body, headers=headers or _headers("admin")
    )


@dataclass(frozen=True)
class _Releases:
    """A seeded two-release lineage plus the versions its tags point at."""

    retriever: LanceDBRetriever
    r1_tag_version: int
    r2_tag_version: int

    @property
    def adapter(self):
        return self.retriever._adapter

    def live_ids(self) -> set[str]:
        """The ids the serving adapter's table holds right now."""
        self.adapter.count()  # opens/keeps the handle the serving path reads
        return set(self.adapter._table.to_pandas()["id"])


@pytest.fixture(scope="module")
def real_main():
    """The app exactly as ``src/main.py`` builds it, seam accessors included.

    Module scope: importing the real app pulls the service layer in (~10s), and
    the app object is the thing the seam is wired onto, so one per module is
    both cheaper and truer than one per test.
    """
    spec = importlib.util.spec_from_file_location("husc_rollback_main", MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def app(real_main, store_root: Path, monkeypatch):
    """The real app, wired over this test's throwaway store root."""
    _configure_identities(monkeypatch)
    configure_canonical_record_store(
        real_main.app, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    return real_main.app


@pytest.fixture
def client(app) -> TestClient:
    """A client that does NOT enter the app's lifespan.

    Entering it loads embedding/reranker models; the rollback route needs the
    routers, the wiring and the seam, none of which live in the lifespan.
    """
    return TestClient(app)


@pytest.fixture
def serving(real_main, monkeypatch, tmp_path: Path) -> LanceDBRetriever:
    """A real LanceDB table, and the process serving from it.

    ``lancedb_retriever_service`` is what ``app._health_lancedb`` resolves, so
    patching it is exactly "this process is serving this table" — the same
    global a query reads.
    """
    _open_table(
        tmp_path,
        [
            _row("r1-a", "Điểm chuẩn 2026 ngành CNTT là 25,5", R1_ID),
            _row("r1-b", "Học phí 2026 ngành CNTT là 15 triệu", R1_ID),
        ],
    )
    retriever = _retriever_over(tmp_path)
    monkeypatch.setattr(real_main, "lancedb_retriever_service", retriever)
    monkeypatch.setattr(real_main, "hybrid_search_service", None)
    return retriever


@pytest.fixture
def releases(serving: LanceDBRetriever, store_root: Path) -> _Releases:
    """Two tagged releases on one table: R2 active, R1 the one behind it.

    The table carries both releases' rows live (R2 was published by adding to
    the same table), so a rollback that restored the tag but not the data would
    still be caught: the live row count has to drop.
    """
    table = serving._adapter._table
    r1_version = tag_release(table, R1_ID)
    table.add([_row("r2-a", "Điểm chuẩn 2026 ngành KTPM là 24,0", R2_ID)])
    r2_version = tag_release(table, R2_ID)

    _seed_release(store_root, R1_ID)
    _seed_release(store_root, R2_ID, parent=R1_ID)
    _activate(store_root, R2_ID)
    return _Releases(
        retriever=serving, r1_tag_version=r1_version, r2_tag_version=r2_version
    )


# ---------------------------------------------------------------------------
# 1. happy path — both levels move, and the response says so
# ---------------------------------------------------------------------------


def test_rollback_moves_canonical_and_retrieval_together(
    client: TestClient, store_root: Path, releases: _Releases
):
    """One request moves the pointer *and* the live table to the previous release."""
    assert _active(store_root) == R2_ID
    assert releases.live_ids() == {"r1-a", "r1-b", "r2-a"}

    response = _post_rollback(client, reason=f"  {REASON}  ")

    assert response.status_code == 200, response.text
    body = response.json()
    # The lineage walked back one hop: R2's parent is R1.
    assert body["release_id"] == R1_ID
    assert body["previous_release_id"] == R2_ID
    assert body["lancedb_tag"] == release_tag(R1_ID)
    assert body["lancedb_tag_version"] == releases.r1_tag_version
    assert body["event_id"]
    # Both levels are reported moved — a half-rollback cannot pass this. The
    # graph is reported too (FIX-02): this app serves no pipeline, so there is
    # no graph to move, and the rollback says so rather than omitting it.
    assert body["steps"] == {
        "validate": "ok",
        "snapshot": "ok",
        "canonical": "ok",
        "retrieval": "ok",
        "graph": "not-serving",
        "audit": "ok",
    }

    # Canonical level: the pointer on disk.
    assert _active(store_root) == R1_ID

    # Retrieval level: the live data, read through the adapter the serving path
    # reads — not the tag, which could point somewhere the table does not serve.
    assert releases.live_ids() == {"r1-a", "r1-b"}
    assert releases.adapter.count() == 2
    # And the tag a later reader treats as "what is in production" moved with it.
    assert releases.adapter._table.tags.list()["prod"]["version"] == releases.r1_tag_version

    # The swap seam republished the app.state alias the meta router reads.
    assert client.app.state.retriever is releases.retriever


def test_the_audit_event_names_the_actor_and_the_trimmed_reason(
    client: TestClient, store_root: Path, releases: _Releases
):
    """The event is the accountability half of the ticket: who, why, from what to what."""
    response = _post_rollback(client, reason=f"  {REASON}  ")
    assert response.status_code == 200, response.text

    events = _rollback_events(store_root)
    assert len(events) == 1
    event = events[0].event
    assert event.actor_id == "cb.quan-tri"
    assert event.role == "admin"
    assert event.reason == REASON  # trimmed, never the padded form
    assert event.target == R1_ID
    assert event.target_kind is TargetKind.RELEASE
    assert event.before_version == R2_ID
    assert event.after_version == R1_ID
    assert event.release_id == R1_ID
    # The response and the ledger agree on which event this was.
    assert response.json()["event_id"] == event.event_id
    assert _ledger(store_root).verify().ok


# ---------------------------------------------------------------------------
# 2. the ledger never records a rollback that did not happen
# ---------------------------------------------------------------------------


def test_no_rollback_event_until_both_levels_have_moved(
    client: TestClient, real_main, store_root: Path, releases: _Releases, monkeypatch
):
    """A step-D failure writes nothing; the same call afterwards writes one event.

    Both halves in one test because the interesting property is the *order*:
    the ledger must stay empty across a failed attempt, and the retry — on the
    state the compensation restored — must produce exactly one event.
    """
    app = real_main.app
    real_seam = app._swap_runtime_services
    mode = {"fail": True}

    def flaky_seam(retriever):
        if mode["fail"]:
            raise RuntimeError("swap seam unavailable")
        return real_seam(retriever)

    monkeypatch.setattr(app, "_swap_runtime_services", flaky_seam)
    ledger = _ledger(store_root)

    failed = _post_rollback(client)
    assert failed.status_code == 500, failed.text
    assert ledger.entries() == ()
    assert _rollback_events(store_root) == ()
    # Nothing was left half-moved either: compensation put both levels back.
    assert _active(store_root) == R2_ID
    assert releases.live_ids() == {"r1-a", "r1-b", "r2-a"}

    mode["fail"] = False
    retried = _post_rollback(client, reason=REASON)
    assert retried.status_code == 200, retried.text

    events = _rollback_events(store_root)
    assert len(events) == 1
    assert events[0].event.reason == REASON
    assert events[0].event.actor_id == "cb.quan-tri"
    assert _ledger(store_root).verify().ok


# ---------------------------------------------------------------------------
# 3. compensation — the two levels must not be able to diverge silently
# ---------------------------------------------------------------------------


def test_step_d_failure_restores_the_pre_call_state(
    client: TestClient, real_main, store_root: Path, releases: _Releases, monkeypatch
):
    """Step C succeeded, step D failed: step E puts both levels back, and says so.

    This is the test that proves a failed rollback is not a half-rollback: the
    pointer it already moved is moved back, and the body tells the operator the
    compensation succeeded rather than leaving them to guess.
    """

    def refuse(retriever):
        raise RuntimeError("swap seam unavailable")

    monkeypatch.setattr(real_main.app, "_swap_runtime_services", refuse)

    response = _post_rollback(client)

    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    # Both halves are named: the retrieval failure, and what compensation did.
    assert "retrieval rollback failed" in detail
    assert "swap seam unavailable" in detail
    assert "compensation succeeded" in detail

    # Canonical: the pointer is back on the release that was active.
    assert _active(store_root) == R2_ID
    # Retrieval: the table serves the pre-call rows again.
    assert releases.live_ids() == {"r1-a", "r1-b", "r2-a"}
    assert releases.adapter.count() == 3
    # And the attempt is not in the ledger — it did not happen.
    assert _rollback_events(store_root) == ()


def test_compensation_failure_is_named_alongside_the_retrieval_failure(
    client: TestClient, real_main, store_root: Path, releases: _Releases, monkeypatch
):
    """When compensation cannot undo the retrieval half, the 5xx must say both.

    The previous release's tag is gone, so the table cannot be restored by tag:
    the canonical pointer is compensated (the operator can still see which
    release is authoritative) while the table stays on the target release. That
    divergence is exactly what the message exists to surface — a body that said
    only "the rollback failed" would let it go unnoticed.
    """

    def refuse(retriever):
        raise RuntimeError("swap seam unavailable")

    monkeypatch.setattr(real_main.app, "_swap_runtime_services", refuse)
    releases.adapter._table.tags.delete(release_tag(R2_ID))

    response = _post_rollback(client)

    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert "retrieval rollback failed" in detail
    assert "COMPENSATION ALSO FAILED" in detail
    assert "restore-back" in detail

    # The half compensation could do: canonical is back on the active release.
    assert _active(store_root) == R2_ID
    # The half it could not: the table still serves the target release's rows.
    assert releases.live_ids() == {"r1-a", "r1-b"}
    assert _rollback_events(store_root) == ()


# ---------------------------------------------------------------------------
# 4. error mapping — one status per failure, driven by real inputs
# ---------------------------------------------------------------------------


def _assert_untouched(store_root: Path, releases: _Releases) -> None:
    """A refused rollback mutates nothing, at either level."""
    assert _active(store_root) == R2_ID
    assert releases.live_ids() == {"r1-a", "r1-b", "r2-a"}
    assert _rollback_events(store_root) == ()


def test_unknown_target_release_is_404(
    client: TestClient, store_root: Path, releases: _Releases
):
    response = _post_rollback(client, target="R-2026-01-01")

    assert response.status_code == 404, response.text
    assert "R-2026-01-01" in response.json()["detail"]
    _assert_untouched(store_root, releases)


def test_target_that_is_already_active_is_409(
    client: TestClient, store_root: Path, releases: _Releases
):
    response = _post_rollback(client, target=R2_ID)

    assert response.status_code == 409, response.text
    assert "already active" in response.json()["detail"]
    _assert_untouched(store_root, releases)


def test_target_without_a_lancedb_tag_is_409(
    client: TestClient, store_root: Path, releases: _Releases
):
    """A release whose chunks were never tagged cannot be restored."""
    _seed_release(store_root, R0_ID, parent=R1_ID)

    response = _post_rollback(client, target=R0_ID)

    assert response.status_code == 409, response.text
    assert release_tag(R0_ID) in response.json()["detail"]
    _assert_untouched(store_root, releases)


def test_manifest_that_disagrees_with_its_directory_is_409(
    client: TestClient, store_root: Path, releases: _Releases
):
    """A manifest naming another release describes a release nobody can reproduce."""
    _seed_release(store_root, "R-2026-06-01", manifest_id="R-2026-06-02")

    response = _post_rollback(client, target="R-2026-06-01")

    assert response.status_code == 409, response.text
    assert "R-2026-06-02" in response.json()["detail"]
    _assert_untouched(store_root, releases)


def test_active_release_with_no_previous_release_is_409(
    client: TestClient, store_root: Path, releases: _Releases
):
    """No target given, and the lineage has nothing to fall back to."""
    _seed_release(store_root, R2_ID, parent=None)

    response = _post_rollback(client)

    assert response.status_code == 409, response.text
    assert "previous release" in response.json()["detail"]
    _assert_untouched(store_root, releases)


@pytest.mark.parametrize(
    "target",
    ["../escape", "a/b", "..", "C:evil", "  "],
    ids=["traversal", "separator", "dotdot", "drive", "blank"],
)
def test_release_id_that_cannot_name_a_directory_is_422(
    client: TestClient, store_root: Path, releases: _Releases, target: str
):
    """The target crosses the HTTP boundary, so it is refused before it is joined."""
    response = _post_rollback(client, target=target)

    assert response.status_code == 422, response.text
    _assert_untouched(store_root, releases)


def test_unreadable_pointer_is_422(
    client: TestClient, store_root: Path, releases: _Releases
):
    _pointer_path(store_root).write_text("{not json", encoding="utf-8")

    response = _post_rollback(client)

    assert response.status_code == 422, response.text
    assert "pointer" in response.json()["detail"]
    # The refusal happened in step A: the table never moved.
    assert releases.live_ids() == {"r1-a", "r1-b", "r2-a"}


def test_no_serving_retriever_is_503(
    client: TestClient, real_main, store_root: Path, monkeypatch
):
    """Warmup has not bound a retriever yet — the request is fine, the runtime is not."""
    _seed_release(store_root, R1_ID)
    _seed_release(store_root, R2_ID, parent=R1_ID)
    _activate(store_root, R2_ID)
    monkeypatch.setattr(real_main, "lancedb_retriever_service", None)

    response = _post_rollback(client)

    assert response.status_code == 503, response.text
    assert "serving retriever" in response.json()["detail"]
    # Nothing was mutated: the pointer is where it was.
    assert _active(store_root) == R2_ID


def test_missing_rollback_service_is_503(client: TestClient, app):
    """An app that never ran the wiring refuses instead of guessing a root."""
    del app.state.admin_rollback_service

    response = _post_rollback(client)

    assert response.status_code == 503, response.text
    assert "admin_rollback_service" in response.json()["detail"]


# ---------------------------------------------------------------------------
# 5. a blank reason is not a reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        pytest.param({"reason": "   "}, id="spaces"),
        pytest.param({"reason": ""}, id="empty"),
        pytest.param({"reason": "\t\n"}, id="control"),
        pytest.param({}, id="missing"),
    ],
)
def test_blank_or_missing_reason_is_refused_before_anything_moves(
    client: TestClient, store_root: Path, releases: _Releases, body: dict
):
    """A rollback nobody can explain is not accountable, so it is refused at the edge."""
    response = client.post("/admin/rollback", json=body, headers=_headers("admin"))

    assert response.status_code == 422, response.text
    _assert_untouched(store_root, releases)


# ---------------------------------------------------------------------------
# 6. RBAC — only the role holding records.rollback may retract live data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "role"),
    [
        (TOKENS["editor"], "editor"),
        (TOKENS["publisher"], "publisher"),
        (OPS_TOKEN, "ops"),
    ],
    ids=["editor", "publisher", "ops"],
)
def test_role_without_the_grant_is_refused_and_changes_nothing(
    client: TestClient, store_root: Path, releases: _Releases, token: str, role: str
):
    response = client.post(
        "/admin/rollback",
        json={"reason": REASON},
        headers={"X-Admin-Token": token},
    )

    assert response.status_code == 403, response.text
    assert "records.rollback" in response.json()["detail"]
    assert role in response.json()["detail"]
    _assert_untouched(store_root, releases)


def test_no_credential_never_reaches_the_handler(
    client: TestClient, store_root: Path, releases: _Releases
):
    """The header is required, so an anonymous caller is stopped before the service."""
    response = client.post("/admin/rollback", json={"reason": REASON})

    assert response.status_code == 422, response.text
    _assert_untouched(store_root, releases)


def test_unknown_credential_is_refused(
    client: TestClient, store_root: Path, releases: _Releases
):
    response = client.post(
        "/admin/rollback",
        json={"reason": REASON},
        headers={"X-Admin-Token": "not-a-registered-credential"},
    )

    assert response.status_code == 403, response.text
    _assert_untouched(store_root, releases)


# ---------------------------------------------------------------------------
# 7. the swap seam — build_hybrid_service is the single construction path
# ---------------------------------------------------------------------------


def test_build_hybrid_service_is_none_when_hybrid_is_disabled(
    real_main, monkeypatch, releases: _Releases
):
    monkeypatch.setattr(real_main.settings, "USE_HYBRID_RETRIEVAL", False)

    assert real_main.build_hybrid_service(releases.retriever) is None


def test_build_hybrid_service_indexes_the_retriever_it_is_given(
    real_main, monkeypatch, releases: _Releases
):
    """The True case, over a real table: a real service, describing that corpus."""
    monkeypatch.setattr(real_main.settings, "USE_HYBRID_RETRIEVAL", True)

    service = real_main.build_hybrid_service(releases.retriever)

    assert isinstance(service, HybridSearchService)
    version = service.corpus_version
    assert version["table"] == TABLE_NAME
    assert version["chunk_count"] == 3  # the live table, both releases' rows


def test_build_hybrid_service_is_none_when_the_bm25_build_fails(
    real_main, monkeypatch, tmp_path: Path
):
    """A corpus with nothing indexable disables the sparse arm rather than serving it."""
    monkeypatch.setattr(real_main.settings, "USE_HYBRID_RETRIEVAL", True)
    _open_table(
        tmp_path,
        [
            _row(
                "s1",
                "Chunk đã bị thay thế bởi bản mới hơn",
                R1_ID,
                metadata={"is_superseded": True},
            )
        ],
    )
    retriever = _retriever_over(tmp_path)

    assert real_main.build_hybrid_service(retriever) is None


def test_a_swap_rebuilds_the_hybrid_service_over_the_new_retriever(
    real_main, monkeypatch, releases: _Releases
):
    """The sparse index always describes the table being served, never the old one."""
    monkeypatch.setattr(real_main.settings, "USE_HYBRID_RETRIEVAL", True)
    stale = object()
    previous = object()
    monkeypatch.setattr(real_main, "hybrid_search_service", stale)
    monkeypatch.setattr(real_main, "lancedb_retriever_service", previous)

    result = real_main.swap_runtime_services(releases.retriever)

    assert result["previous_retriever"] is previous
    assert result["hybrid_rebuilt"] is True
    assert real_main.lancedb_retriever_service is releases.retriever
    assert real_main.hybrid_search_service is not stale
    assert isinstance(real_main.hybrid_search_service, HybridSearchService)


def test_a_failed_hybrid_rebuild_drops_the_stale_index(
    real_main, monkeypatch, tmp_path: Path
):
    """A build that fails must not leave the *old* table's index serving queries."""
    monkeypatch.setattr(real_main.settings, "USE_HYBRID_RETRIEVAL", True)
    stale = object()
    monkeypatch.setattr(real_main, "hybrid_search_service", stale)
    _open_table(
        tmp_path,
        [
            _row(
                "s1",
                "Chunk đã bị thay thế bởi bản mới hơn",
                R1_ID,
                metadata={"is_superseded": True},
            )
        ],
    )
    retriever = _retriever_over(tmp_path)

    result = real_main.swap_runtime_services(retriever)

    assert result["hybrid_rebuilt"] is False
    assert real_main.hybrid_search_service is None

# ---------------------------------------------------------------------------
# 8. the drain gate — a timeout is a refusal, not a licence to move under load
#
# A rollback retags and rebinds the table every query reads, so the drain exists
# to prove nobody is reading the release being rolled away from. "Log the
# timeout and roll back anyway" inverted that: it moved the release under a
# request that was already reading it. These two tests hold a request in flight
# for real and assert what the gate does with it.
# ---------------------------------------------------------------------------

def _admin_route_globals(real_main, path: str) -> dict:
    """The namespace the admin route reads its module constants from.

    ``routers.admin`` and ``src.routers.admin`` are two *distinct* module objects
    in this suite (``conftest.py`` puts both ``rag2025`` and ``rag2025/src`` on
    ``sys.path``), and the route mounted on the app belongs to whichever one
    ``src/main.py`` imported. Reading the namespace off the route's own endpoint
    is the only way to patch the constant the handler will actually look up.
    """
    for route in real_main.admin_router.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint.__globals__
    raise AssertionError(f"no route registered at {path}")

def _hold_one_request_in_flight(tracker):
    """Hold ``tracker``'s counter at 1 until the returned release callable runs.

    A real ``track()`` section, entered from a thread with its own event loop and
    parked inside it. So the drain that times out is expiring against the counter
    an actual request holds — the same state the route has to survive — rather
    than against a stubbed ``drain()`` that returns False.
    """
    gate = threading.Event()
    entered = threading.Event()

    async def _hold() -> None:
        async with tracker.track():
            entered.set()
            await asyncio.to_thread(gate.wait)

    thread = threading.Thread(target=lambda: asyncio.run(_hold()), daemon=True)
    thread.start()
    assert entered.wait(timeout=10.0), "the held request never became in-flight"
    assert tracker.in_flight == 1
    return gate.set, thread

def test_rollback_refuses_on_drain_timeout_and_moves_nothing(
    client: TestClient,
    app,
    real_main,
    store_root: Path,
    releases: _Releases,
    monkeypatch,
):
    """A rollback whose drain times out is refused, with the release still serving.

    Held for real, so ``drain()`` genuinely expires. The refusal must hold on
    four points, in order of what the old branch broke:

    1. 409, and the message names the override — not a 200 that quietly moved;
    2. neither level moved: pointer, live table and ledger untouched;
    3. the retriever every query reads is still the one that was serving, and it
       is still queryable through the adapter the serving path uses;
    4. admission reopened, so the refusal did not leave the process refusing
       every query.
    """
    monkeypatch.setitem(
        _admin_route_globals(real_main, "/admin/rollback"), "DEFAULT_DRAIN_TIMEOUT_S", 0.3
    )
    tracker = type(app.state.request_tracker)()
    monkeypatch.setattr(app.state, "request_tracker", tracker)
    retriever_before = real_main.lancedb_retriever_service

    release, thread = _hold_one_request_in_flight(tracker)
    try:
        response = _post_rollback(client)
    finally:
        release()
        thread.join(timeout=10.0)

    # 1. Refused, with the way out named.
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert "drain timed out" in detail, detail
    assert "force_while_in_flight=true" in detail, detail
    assert response.headers.get("Retry-After") == "1", dict(response.headers)

    # 2. Nothing moved, at either level.
    _assert_untouched(store_root, releases)

    # 3. The runtime every query reads was not rebound...
    assert real_main.lancedb_retriever_service is retriever_before
    # ...and the release it refused to abandon still answers, read through the
    # adapter the serving path itself uses.
    assert releases.live_ids() == {"r1-a", "r1-b", "r2-a"}

    # 4. Admission reopened on the way out of the refusal.
    assert tracker.admitting is True
    assert tracker.in_flight == 0

def test_forced_rollback_moves_under_load_when_the_caller_says_so(
    client: TestClient,
    app,
    real_main,
    store_root: Path,
    releases: _Releases,
    monkeypatch,
):
    """``force_while_in_flight`` is the emergency, and it is a named one.

    The flag has to be real — a forced rollback actually moves both levels — and
    the response has to say it was forced, because that is the operator's only
    signal that a reader may have been cut off mid-answer.
    """
    monkeypatch.setitem(
        _admin_route_globals(real_main, "/admin/rollback"), "DEFAULT_DRAIN_TIMEOUT_S", 0.3
    )
    tracker = type(app.state.request_tracker)()
    monkeypatch.setattr(app.state, "request_tracker", tracker)

    release, thread = _hold_one_request_in_flight(tracker)
    try:
        response = client.post(
            "/admin/rollback",
            json={"reason": REASON, "force_while_in_flight": True},
            headers=_headers("admin"),
        )
    finally:
        release()
        thread.join(timeout=10.0)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["forced_while_in_flight"] is True, body
    assert body["release_id"] == R1_ID, body
    assert body["steps"]["retrieval"] == "ok", body

    # Both levels moved, exactly as an unforced rollback moves them.
    assert _active(store_root) == R1_ID
    assert releases.live_ids() == {"r1-a", "r1-b"}
    assert len(_rollback_events(store_root)) == 1
    assert tracker.admitting is True

# ===========================================================================
# Ticket 22 / FIX-02 — the graph is part of the release unit
#
# The graph maps entity nodes to the `chunk_id`s they came from. A graph from
# release R2 answering over release R1's table injects R2's chunk ids into an
# R1 answer — a mixed-release answer built from a *matching* pointer and table,
# which the LanceDB half alone cannot detect. These tests pin both artifacts and
# require the rollback to move both.
# ===========================================================================

def _graph_with(chunk_id: str, entity: str) -> KnowledgeGraph:
    """A one-node graph whose entity node names ``chunk_id``.

    Small on purpose: PPR over a single node still resolves to the chunk ids
    the node carries, which is the whole question — "which release's chunk ids
    does the served graph resolve to?" — asked of a real graph rather than of a
    recorded variable.
    """
    import networkx as nx

    g = nx.MultiDiGraph()
    g.graph["schema_version"] = GRAPH_SCHEMA_VERSION
    g.add_node(
        entity,
        text=entity,
        entity_type="NGANH",
        normalized=entity,
        chunk_ids=json.dumps([chunk_id]),
    )
    return KnowledgeGraph(g)

def _pin_release_graph(store_root: Path, release_id: str, graph: KnowledgeGraph) -> str:
    """Publish ``graph`` into ``release_id`` — the real publish-side call.

    Writes the built graph outside the release (as a build does), then pins it
    through :func:`publish_release_graph`, the one call a publish path makes for
    the graph half of a release. The pin therefore lands in the manifest by the
    same code path production uses, not by the test writing manifest fields.
    """
    build = store_root / f"_build_{release_id}.graphml"
    build.parent.mkdir(parents=True, exist_ok=True)
    graph.save(build)
    manifest = publish_release_graph(_releases_dir(store_root) / release_id, build)
    return manifest.graph_sha256

class _ServingGraphRAG:
    """The graphrag half of a pipeline, as the rollback reaches it."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graph = graph
        self.swaps: list[KnowledgeGraph] = []

    def update_graph(self, new_graph: KnowledgeGraph) -> None:
        self._graph = new_graph
        self.swaps.append(new_graph)

class _ServingPipeline:
    """A pipeline serving a graph — the object ``app._health_pipeline`` yields."""

    def __init__(self, graph: KnowledgeGraph) -> None:
        self._graphrag = _ServingGraphRAG(graph)

@pytest.fixture
def serving_graph(real_main, monkeypatch):
    """The process serving R2's graph, as ``_health_pipeline`` resolves it.

    Patched on the main module because ``app._health_pipeline`` is
    ``lambda: unified_pipeline`` — the same global a query path reads.
    """
    graph = _graph_with("r2-a", "NGANH:ky_thuat_phan_mem")
    pipeline = _ServingPipeline(graph)
    monkeypatch.setattr(real_main, "unified_pipeline", pipeline)
    return pipeline

@pytest.fixture
def graph_releases(store_root: Path, releases: _Releases, serving_graph) -> _Releases:
    """The same R2-active lineage, with both releases pinning a graph."""
    _pin_release_graph(store_root, R1_ID, _graph_with("r1-a", "NGANH:cong_nghe_thong_tin"))
    _pin_release_graph(store_root, R2_ID, _graph_with("r2-a", "NGANH:ky_thuat_phan_mem"))
    return releases

def _served_chunk_ids(real_main) -> set[str]:
    """The chunk ids the graph the process serves resolves to — by PPR.

    A real personalized-PageRank query over the served graph object, aggregated
    to chunk level: exactly what a PPR-only query path does. Not a check of a
    recorded variable — of the graph that would answer.
    """
    graph = real_main.unified_pipeline._graphrag._graph
    seeds = [n for n in graph._graph.nodes]
    scores = asyncio.run(graph.ppr_scores_by_chunk(seeds))
    return set(scores)

def test_rollback_moves_the_graph_too_or_the_pairing_is_wrong(
    client: TestClient,
    real_main,
    store_root: Path,
    graph_releases: _Releases,
):
    """THE CORE TEST: a graph left on R2 while the table moves to R1 fails.

    Before the graph was part of the release unit this test fails — the served
    graph keeps resolving to R2's chunk ids, so a PPR-only query after the
    rollback answers from the release that was rolled away from.
    """
    assert _active(store_root) == R2_ID
    assert _served_chunk_ids(real_main) == {"r2-a"}

    response = _post_rollback(client, reason=REASON)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["steps"]["graph"] == "ok", body

    # The table moved...
    assert graph_releases.live_ids() == {"r1-a", "r1-b"}
    # ...and so did the graph, proven by a PPR-only query over the graph the
    # process actually serves, not by the value of a variable.
    assert _served_chunk_ids(real_main) == {"r1-a"}, (
        "the served graph still resolves to the previous release's chunk ids"
    )
    # The response names the artifact, so an operator can tell which bytes are
    # being served without reading the pipeline.
    assert body["steps"]["graph"] == "ok"
    manifest = read_manifest(_releases_dir(store_root) / R1_ID)
    assert real_main.unified_pipeline._graphrag._graph is not None
    assert manifest.graph_sha256
    assert manifest.graph_schema_version == GRAPH_SCHEMA_VERSION
    assert len(_rollback_events(store_root)) == 1

def test_a_target_that_pins_no_graph_is_refused_over_a_serving_graph(
    client: TestClient,
    real_main,
    store_root: Path,
    releases: _Releases,
    serving_graph,
):
    """An unpinned target over a serving graph is a 4xx with nothing mutated.

    The service's own reports are the honest ones here: the running graph would
    survive the rollback and answer the target's table with the previous
    release's chunk ids, so the request is refused before anything moves.
    """
    assert read_manifest(_releases_dir(store_root) / R1_ID).graph_sha256 == ""

    response = _post_rollback(client, reason=REASON)

    assert response.status_code == 422, response.text
    assert "pins no graph artifact" in response.json()["detail"]
    _assert_untouched(store_root, releases)
    assert _served_chunk_ids(real_main) == {"r2-a"}

def test_a_rebuilt_pinned_graph_is_refused_before_anything_moves(
    client: TestClient,
    real_main,
    store_root: Path,
    graph_releases: _Releases,
):
    """The pin is verified, not trusted: bytes that no longer match it refuse.

    This is the "rebuilt in place after the release was published" case — the
    failure the digest exists to catch. It must refuse with the pointer, the
    table and the served graph all untouched.
    """
    artifact = _releases_dir(store_root) / R1_ID / "graph" / "knowledge_graph.graphml"
    assert artifact.is_file()
    rebuilt = _graph_with("r1-zzz", "NGANH:cong_nghe_thong_tin")
    rebuilt.save(artifact)

    response = _post_rollback(client, reason=REASON)

    assert response.status_code == 422, response.text
    assert "not restorable" in response.json()["detail"], response.json()
    _assert_untouched(store_root, graph_releases)
    assert _served_chunk_ids(real_main) == {"r2-a"}

def test_the_graph_step_is_reported_when_no_graph_is_served(
    client: TestClient, store_root: Path, releases: _Releases
):
    """A runtime that serves no graph reports it, and still rolls back.

    No pipeline in this test: ``unified_pipeline`` is None, which is the state
    of a deployment without GraphRAG. Rolling back must work there — refusing
    would make rollback impossible exactly when it is needed — and the absence
    must be *reported* rather than silently omitted.
    """
    response = _post_rollback(client, reason=REASON)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["steps"]["graph"] == "not-serving", body
    assert _active(store_root) == R1_ID
    assert releases.live_ids() == {"r1-a", "r1-b"}

def test_a_graph_that_cannot_be_applied_compensates_the_table_and_pointer(
    client: TestClient,
    real_main,
    store_root: Path,
    graph_releases: _Releases,
    monkeypatch,
):
    """Step D2 failing rolls the table and the pointer back, and says so.

    The failure is injected at the pipeline (the graphrag refusing the new
    graph), which is where a real one would come from. Nothing may be left
    serving the target's table with the previous graph.
    """
    from src.services.admin_rollback import ReleaseGraphRollback

    real_update = ReleaseGraphRollback.apply

    def _refuse(self, manifest, release_dir):
        raise GraphRollbackErrorForTest("injected: the runtime refused the graph")

    monkeypatch.setattr(ReleaseGraphRollback, "apply", _refuse)

    response = _post_rollback(client, reason=REASON)

    assert response.status_code == 500, response.text
    detail = response.json()["detail"]
    assert "stage 'graph'" in detail, detail
    assert "graph rollback failed" in detail, detail

    # Compensated: the pointer is back on R2 and the table serves R2 again.
    assert _active(store_root) == R2_ID
    assert graph_releases.live_ids() == {"r1-a", "r1-b", "r2-a"}
    # The graph never moved, so it still resolves to R2's chunk ids — and the
    # ledger records no rollback that did not happen.
    assert _served_chunk_ids(real_main) == {"r2-a"}
    assert _rollback_events(store_root) == ()
    assert real_update is not None

class GraphRollbackErrorForTest(Exception):
    """A graph failure injected at the pipeline, for the compensation test."""
