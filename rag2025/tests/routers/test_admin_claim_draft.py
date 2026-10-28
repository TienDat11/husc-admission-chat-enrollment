"""T15: claim a draft (nhận nháp) — the seed-draft four-eyes fix.

The seed drafts (spec 02 §108) all carry ``authored_by = null``, and
four-eyes refuses to approve a version whose author is not recorded — so
"duyệt cả nhóm" on a seed draft is impossible until someone claims it. A
claim is a new version, not a rewrite: the seed version and its audit
history stay where they were, the new version is the one the claimer
authored, and it can now move through submit -> approve by a *different*
person.

The two acceptance criteria, asserted at the HTTP seam (the same seam
ticket 12's rule names):

(a) approving an unclaimed draft is refused — four-eyes, 403 — while the
    claim is the way to make it approvable;
(b) claim + submit + approve by a different person succeeds.

Refusals leave the store untouched: a claim on a version that is not an
unclaimed draft is a 409 and writes nothing, the way every other lifecycle
refusal does.
"""
from __future__ import annotations

import json
import sys
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
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.canonical_record_store import (  # noqa: E402
    CanonicalRecordStore,
)
from src.routers.admin import router as admin_router  # noqa: E402

# One actor per role, mirroring ticket 13's registry shape: a credential
# presents, the server decides the role; nothing in a request names one.
ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "editor": ("cb.bien-tap", "editor"),
    "reviewer": ("cb.phan-bien", "reviewer"),
    "publisher": ("cb.xuat-ban", "publisher"),
    "viewer": ("cb.chi-xem", "viewer"),
}
SECOND_REVIEWER = ("cb.phan-bien-2", "reviewer")

TOKENS = {role: f"tok-{role}" for role in ACTORS}

RECORD_ID = "DIEM_CHUAN:NGANH:cntt:2026"
TEXT_VI = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là 25,5 điểm."


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _configure(monkeypatch) -> None:
    entries = [
        {"token": TOKENS[role], "actor_id": actor_id, "role": role}
        for role, (actor_id, _) in ACTORS.items()
    ]
    entries.append(
        {"token": "tok-reviewer-2", "actor_id": SECOND_REVIEWER[0], "role": SECOND_REVIEWER[1]}
    )
    monkeypatch.setenv("ADMIN_IDENTITIES", json.dumps(entries))
    monkeypatch.setenv("ADMIN_API_TOKEN", "shared-ops-token")


def _seed_draft(store_root: Path) -> None:
    """Write one seed-shaped draft straight to the store: version 1, ``draft``,
    ``authored_by = null`` — the shape spec 02 §108 says the 487 seeds carry.

    A seed arrives through the ingestion script, not the admin UI, so the test
    seeds it the way the seed would: the store write is the seed, and the
    claim is what the office does next."""
    store = CanonicalRecordStore(store_root)
    store.write_record(
        CanonicalRecord(
            record_id=RECORD_ID,
            record_type=RecordType.DIEM_CHUAN,
            version=1,
            admission_year=2026,
            effective_from="2026-06-01",
            status=RecordStatus.DRAFT,
            authority_level=AuthorityLevel.HUSC_OFFICIAL,
            source_document_id="de-an-tuyen-sinh-2026",
            text=TEXT_VI,
        )
    )


def _claim(client: TestClient, role: str, *, expected_version: int | None = None, reason: str | None = None):
    body: dict = {}
    if expected_version is not None:
        body["expected_version"] = expected_version
    if reason is not None:
        body["reason"] = reason
    return client.post(
        f"/admin/records/{RECORD_ID}/claim", json=body, headers=_headers(role)
    )


def _transition(client: TestClient, role: str, action: str, *, expected_version: int | None = None, reason: str | None = None):
    body: dict = {"action": action}
    if expected_version is not None:
        body["expected_version"] = expected_version
    if reason is not None:
        body["reason"] = reason
    return client.post(
        f"/admin/records/{RECORD_ID}/transitions", json=body, headers=_headers(role)
    )


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def app(store_root: Path, monkeypatch) -> FastAPI:
    _configure(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def disk(store_root: Path) -> CanonicalRecordStore:
    """A second, independent store over the same root — reads real bytes."""
    return CanonicalRecordStore(store_root)


# ---------------------------------------------------------------------------
# (a) approving an unclaimed draft is refused; claim + submit + approve works
# ---------------------------------------------------------------------------


def test_approving_an_unclaimed_seed_draft_is_refused(client, disk, store_root):
    """Criterion (a): no recorded author means no approvable version.

    The seed sits in draft with ``authored_by = null``. An approve of it is
    impossible even before four-eyes — approve is not legal from draft at
    all — so the draft must first be submitted; the review is where the
    missing authorship shows up, and it shows up as a 403 that says the
    author is not recorded, not a silent pass.
    """
    _seed_draft(store_root)
    assert disk.get_record(RECORD_ID).latest.authored_by is None

    assert _transition(client, "editor", "submit", expected_version=1).status_code == 200
    before = disk.versions(RECORD_ID)

    refused = _transition(client, "reviewer", "approve", expected_version=2)

    assert refused.status_code == 403, refused.text
    assert "author" in refused.json()["detail"]
    # The refusal is a no-op: still in review, nothing written.
    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "review"


def test_claim_submit_approve_makes_the_seed_draft_approvable(client, disk, store_root):
    """Criterion (b): the claim, then the review a different person makes.

    The claim appends version 2 — the seed version 1 and its history stay
    exactly where they were — and is the version whose ``authored_by`` the
    four-eyes check finally has something to compare the approver against.
    """
    _seed_draft(store_root)

    claimed = _claim(client, "editor", expected_version=1, reason="nhận nháp seed")
    assert claimed.status_code == 200, claimed.text
    new_version = claimed.json()["record"]
    assert new_version["version"] == 2
    assert new_version["status"] == "draft"
    assert new_version["authored_by"] == ACTORS["editor"][0]
    # The claim says what it did and who did it.
    event = claimed.json()["event"]
    assert event["action"] == "claim"
    assert event["actor_id"] == ACTORS["editor"][0]
    assert event["before_status"] == "draft"
    assert event["after_status"] == "draft"

    # The seed version is intact on disk: null author, the original bytes.
    seed = disk.get_version(RECORD_ID, 1)
    assert seed.authored_by is None
    assert seed.text == TEXT_VI
    # …and it is still in the history the office can read.
    assert disk.versions(RECORD_ID) == (1, 2)

    assert _transition(client, "editor", "submit", expected_version=2).status_code == 200
    approved = _transition(client, "reviewer", "approve", expected_version=3)
    assert approved.status_code == 200, approved.text
    assert approved.json()["record"]["status"] == "approved"
    assert approved.json()["event"]["actor_id"] == ACTORS["reviewer"][0]
    # The claim is what the approval rested on: version 2 is the author's.
    assert disk.get_version(RECORD_ID, 2).authored_by == ACTORS["editor"][0]


# ---------------------------------------------------------------------------
# claim refusals: only an unclaimed draft may be claimed, and a refusal
# writes nothing
# ---------------------------------------------------------------------------


def test_claiming_a_draft_someone_already_claimed_is_refused(client, disk, store_root):
    """Re-claiming is not a second claim: the version already has an author,
    and swapping it for a different one is an edit, not a claim."""
    _seed_draft(store_root)
    assert _claim(client, "editor", expected_version=1).status_code == 200
    before = disk.versions(RECORD_ID)

    # Your own draft again — nothing new to say.
    own = _claim(client, "editor", expected_version=2)
    assert own.status_code == 409, own.text
    # Someone else's draft — claiming is not a re-authoring.
    stolen = _claim(client, "admin", expected_version=2)
    assert stolen.status_code == 409, stolen.text

    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.authored_by == ACTORS["editor"][0]


def test_claiming_a_record_in_a_decided_status_is_refused(client, disk, store_root):
    """Claiming a reviewed/approved/published/archived version would demote a
    decision back to draft — the same move an edit performs openly, never
    silently. A claim is only legal on an unclaimed draft."""
    _seed_draft(store_root)
    assert _claim(client, "editor", expected_version=1).status_code == 200
    assert _transition(client, "editor", "submit", expected_version=2).status_code == 200
    before = disk.versions(RECORD_ID)

    refused = _claim(client, "admin", expected_version=3)

    assert refused.status_code == 409, refused.text
    assert "status" in refused.json()["detail"]
    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "review"

    # Approved and published, each still a 409.
    assert _transition(client, "reviewer", "approve", expected_version=3).status_code == 200
    before = disk.versions(RECORD_ID)
    refused = _claim(client, "admin", expected_version=4)
    assert refused.status_code == 409, refused.text
    assert disk.versions(RECORD_ID) == before

    assert _transition(client, "publisher", "publish", expected_version=4).status_code == 200
    before = disk.versions(RECORD_ID)
    refused = _claim(client, "admin", expected_version=5)
    assert refused.status_code == 409, refused.text
    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.status == "published"


def test_claims_without_the_edit_grant_are_refused(client, disk, store_root):
    """Claim is an editor's grant (RECORDS_EDIT); a viewer or publisher
    credential does not carry it — the refusal happens before the store
    is touched, like every other route-level denial."""
    _seed_draft(store_root)
    before = disk.versions(RECORD_ID)

    for role in ("viewer", "publisher"):
        refused = _claim(client, role, expected_version=1)
        assert refused.status_code == 403, f"{role} -> {refused.text}"

    assert disk.versions(RECORD_ID) == before
    assert disk.get_record(RECORD_ID).latest.authored_by is None
