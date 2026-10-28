"""TDD: GET /admin/releases and GET /admin/index-jobs — ticket 26's read APIs.

Asserted at the HTTP seam against the app ``src/main.py`` builds, wired over a
throwaway canonical-store root, because both routes resolve their subject from
``app.state`` — the releases root from ``admin_rollback_service`` and the job
sources from ``reindex_outbox``/``reindex_job``. A hand-mounted router would
answer a different question than the deployed one.

What is covered, and why each is here:

- the release route reports the active release and the one behind it, and
  refuses (503) when the app has no rollback service rather than guessing a root;
- an unreadable pointer is a *problem*, never rendered as "chưa publish" — the
  distinction a screen whose whole job is naming the live release cannot lose;
- the job route reports pending/running/failed/active from a live job, and a
  failed job carries its reason;
- **saved is never published**: a build in the staging area is
  ``saved=true, serving=false``, its label does not carry the active/published
  wording, and no row of the response claims to serve — asserted over the raw
  JSON body, so it holds for whatever the UI decides to render;
- both routes are read-only GETs under ``records.read``: a role without the
  grant is refused, and neither route mutates the pointer or the staging area.

Isolation: every test gets its own ``tmp_path`` root; nothing touches the
developer's real corpus. No network, no LLM.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.domain.release import ReleaseManifest  # noqa: E402
from src.infrastructure.canonical_store import (  # noqa: E402
    publish_pointer,
    write_manifest,
)
from src.services.admin_rollback import (  # noqa: E402
    POINTER_FILENAME,
    RELEASES_DIRNAME,
    release_tag,
)
from src.services.reindex_job import (  # noqa: E402
    CHUNK_MANIFEST_FILENAME,
    CorpusStamp,
    JobState,
)

MAIN_PATH = ROOT / "src" / "main.py"

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "publisher": ("cb.xuat-ban", "publisher"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}

R1_ID = "R-2026-08-01"
R2_ID = "R-2026-09-01"


def _headers(role: str = "admin") -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


@pytest.fixture(scope="module")
def real_main():
    """The app exactly as ``src/main.py`` builds it. Module scope: the import
    pulls the whole service layer in (~10s) and the app is what is under test."""
    spec = importlib.util.spec_from_file_location("husc_release_status_main", MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def app(real_main, store_root: Path, monkeypatch):
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": TOKENS[role], "actor_id": actor, "role": role}
                for role, (actor, _) in ACTORS.items()
            ]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", "shared-ops-token")
    configure_canonical_record_store(
        real_main.app, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    return real_main.app


@pytest.fixture
def client(app) -> TestClient:
    """A client that does not enter the lifespan — neither route needs a model."""
    return TestClient(app)


def _releases_dir(store_root: Path) -> Path:
    return store_root / RELEASES_DIRNAME


def _seed(store_root: Path, release_id: str, *, parent: str | None = None) -> None:
    write_manifest(
        _releases_dir(store_root) / release_id,
        ReleaseManifest(
            release_id=release_id,
            parent_release_id=parent,
            lancedb_tag=release_tag(release_id),
        ),
    )


def _activate(store_root: Path, release_id: str) -> None:
    publish_pointer(_releases_dir(store_root) / POINTER_FILENAME, release_id)


def _staged_dir(tmp_path: Path, name: str) -> Path:
    """A build in the staging area of this test's own corpus dir."""
    directory = tmp_path / "canonical_corpus.staging" / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CHUNK_MANIFEST_FILENAME).write_text(
        json.dumps(CorpusStamp.current().to_dict()), encoding="utf-8"
    )
    return directory


# ---------------------------------------------------------------------------
# GET /admin/releases


def test_the_release_route_names_the_active_release_and_the_one_behind_it(
    client: TestClient, store_root: Path
):
    _seed(store_root, R1_ID)
    _seed(store_root, R2_ID, parent=R1_ID)
    _activate(store_root, R2_ID)

    response = client.get("/admin/releases", headers=_headers())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["active_release_id"] == R2_ID
    assert body["previous_release_id"] == R1_ID
    assert body["active"]["release_id"] == R2_ID
    assert body["previous"]["release_id"] == R1_ID
    assert body["active"]["lancedb_tag"] == release_tag(R2_ID)
    assert body["problems"] == []


def test_the_release_route_reports_nothing_published_before_the_first_pointer(
    client: TestClient,
):
    """A fresh store has no active release — and that is not an error."""
    response = client.get("/admin/releases", headers=_headers())

    assert response.status_code == 200
    assert response.json()["active_release_id"] is None
    assert response.json()["problems"] == []


def test_a_corrupt_pointer_is_a_problem_not_a_silent_no_release(
    client: TestClient, store_root: Path
):
    """The one answer this screen must never give wrongly."""
    root = _releases_dir(store_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / POINTER_FILENAME).write_text("<<not json>>", encoding="utf-8")

    body = client.get("/admin/releases", headers=_headers()).json()

    assert body["active_release_id"] is None
    assert body["problems"], "a corrupt pointer must be reported"
    assert "not valid JSON" in body["problems"][0]


def test_the_release_route_refuses_when_no_rollback_service_is_wired(
    real_main, monkeypatch, client: TestClient
):
    """503, not a guessed root: the releases root is the rollback's, or unknown.

    The ``app`` fixture re-wires the service for every test, so deleting it here
    cannot leak into another test.
    """
    monkeypatch.delattr(real_main.app.state, "admin_rollback_service", raising=False)

    response = client.get("/admin/releases", headers=_headers())

    assert response.status_code == 503
    assert "admin_rollback_service" in response.json()["detail"]


def test_the_release_route_refuses_an_unknown_credential(client: TestClient):
    """403 for an unknown credential — the code every other admin route gives."""
    response = client.get("/admin/releases", headers={"X-Admin-Token": "tok-nobody"})

    assert response.status_code == 403


def test_the_release_route_refuses_a_request_with_no_credential(client: TestClient):
    """422: the credential header is required, so "no identity" is a 422."""
    assert client.get("/admin/releases").status_code == 422


# ---------------------------------------------------------------------------
# GET /admin/index-jobs


def test_the_job_route_reports_the_four_states_from_a_live_job(
    client: TestClient, app, monkeypatch
):
    """pending/running/failed/active, each with its own label and reason."""

    class _Job:
        def __init__(self, state: JobState, error=None):
            self.state = state
            self.saved = True
            self.serving = state.is_serving
            self.staged_dir = None
            self.error = error

    cases = [
        (JobState.PENDING, None, "chờ chạy (pending)"),
        (JobState.RUNNING, None, "đang chạy (running)"),
        (JobState.FAILED, RuntimeError("provider hết quota"), "thất bại (failed)"),
        (JobState.ACTIVE, None, "đang phục vụ (active)"),
    ]
    for state, error, label in cases:
        monkeypatch.setattr(app.state, "reindex_job", _Job(state, error), raising=False)
        body = client.get("/admin/index-jobs", headers=_headers()).json()
        live = body["jobs"][0]
        assert live["state"] == state.value
        assert live["label"] == label
        if state is JobState.FAILED:
            assert live["reason"] == "provider hết quota"
        if state is not JobState.ACTIVE:
            assert live["serving"] is False


def test_a_failed_job_carries_its_reason_and_a_retry(client: TestClient, app, monkeypatch):
    """The screen's "vì sao hỏng" and "retry" answers, from the route."""

    class _FailedJob:
        state = JobState.FAILED
        saved = True
        serving = False
        staged_dir = None
        error = RuntimeError("embedder trả 403 insufficient quota")

    monkeypatch.setattr(app.state, "reindex_job", _FailedJob(), raising=False)

    body = client.get("/admin/index-jobs", headers=_headers()).json()
    live = body["jobs"][0]

    assert "403 insufficient quota" in live["reason"]
    steps = {step["step"]: step for step in live["retry"]}
    assert set(steps) == {"rebuild", "ingest"}
    assert steps["rebuild"]["runnable"] is False
    assert steps["rebuild"]["reason"] is None


def test_a_saved_build_is_reported_saved_and_never_as_published(
    client: TestClient, tmp_path: Path, monkeypatch
):
    """Ticket 26 criterion 3, asserted over the response body itself.

    The service's default corpus dir is pointed at this test's ``tmp_path``, so
    the real route reads a real staged build with no fake in the path. The build
    is ``saved`` — and nothing in the body says it serves, is active, or is
    published.
    """
    from src.services import release_status as rs

    corpus = tmp_path / "canonical_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    staged = _staged_dir(tmp_path, "job-20260913T000000000000")
    monkeypatch.setattr(rs, "DEFAULT_CORPUS_DIR", corpus)

    body = client.get("/admin/index-jobs", headers=_headers()).json()

    row = next(r for r in body["jobs"] if r["job_id"] == staged.name)
    assert row["saved"] is True
    assert row["serving"] is False
    assert row["state"] is None
    assert "chưa phục vụ" in row["label"]
    assert "đang phục vụ" not in row["label"]
    assert "active" not in row["label"].lower()
    # And over the whole payload: no row claims to serve while no job is active.
    assert all(r["serving"] is False for r in body["jobs"])


def test_the_job_route_answers_with_nothing_wired(
    client: TestClient, tmp_path: Path, monkeypatch
):
    """The deployed reality today (no runner, empty outbox): an honest empty list.

    The staging root is pointed at an empty ``tmp_path``: the route otherwise
    reads the developer's real ``data/canonical_corpus.staging``, so any build
    left there by a live reindex made this test depend on the machine.
    """
    from src.services import release_status as rs

    corpus = tmp_path / "canonical_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(rs, "DEFAULT_CORPUS_DIR", corpus)

    body = client.get("/admin/index-jobs", headers=_headers()).json()

    assert body["jobs"] == []
    assert body["pending_outbox_entries"] == 0
    assert body["problems"] == []
    assert isinstance(body["reindex_needed"], bool)


def test_the_job_route_refuses_an_unknown_credential(client: TestClient):
    """Same auth surface as every other admin route: 403 unknown, 422 absent."""
    assert client.get("/admin/index-jobs", headers={"X-Admin-Token": "tok-nobody"}).status_code == 403
    assert client.get("/admin/index-jobs").status_code == 422


# ---------------------------------------------------------------------------
# read-only: neither route moves anything


def test_neither_route_mutates_the_pointer_or_the_staging_area(
    client: TestClient, store_root: Path, tmp_path: Path, monkeypatch
):
    """Proven by bytes: same pointer, same staging listing, before and after."""
    from src.services import release_status as rs

    _seed(store_root, R1_ID)
    _seed(store_root, R2_ID, parent=R1_ID)
    _activate(store_root, R2_ID)
    corpus = tmp_path / "canonical_corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    _staged_dir(tmp_path, "job-20260913T000000000000")
    monkeypatch.setattr(rs, "DEFAULT_CORPUS_DIR", corpus)

    pointer = _releases_dir(store_root) / POINTER_FILENAME
    before_pointer = pointer.read_bytes()
    before_staging = sorted(p.name for p in (tmp_path / "canonical_corpus.staging").iterdir())

    assert client.get("/admin/releases", headers=_headers()).status_code == 200
    assert client.get("/admin/index-jobs", headers=_headers()).status_code == 200

    assert pointer.read_bytes() == before_pointer
    after_staging = sorted(p.name for p in (tmp_path / "canonical_corpus.staging").iterdir())
    assert after_staging == before_staging
