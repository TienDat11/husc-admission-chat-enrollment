"""TDD: /admin/conflicts — ticket 38.

Asserted at the HTTP seam (per the spec's "mọi test admin chạy qua HTTP
layer"), on a hand-mounted app over a throwaway store root — the same shape
``test_admin_lifecycle_api.py`` uses for its behavioral cases:

- criterion 1: GET lists the seed report's conflicts, including the
  7220104/ten_nganh/2026 pair with exactly its two distinct values;
- criterion 3: a resolve without a non-blank reason is refused 422 and
  records nothing;
- criterion 4: a full resolve emits one proper ``conflict_override`` event —
  actor from the credential, never the body — points at the winning version,
  flips the item to ``resolved``, and a second resolve answers 409;
- criterion 6: the resolve route is gated by the override grant (viewer 403,
  missing credential 422);
- criterion 9: the response states total / seed-reported / deduped counts.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.domain.audit import AuditAction  # noqa: E402
from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402

# Ticket 13: identity is a credential -> (actor_id, role) registry, configured
# per deployment. One actor per role so a test can name the actor it means.
ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}
OPS_TOKEN = "shared-ops-token"


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


WINNER_DOC = "new_data/THONG_TIN_TUYEN_SINH_202611.pdf"


def _pair_item(payload: dict, record_type: str = "NGANH") -> dict:
    """The criterion-1 pair as the API returned it."""
    items = [
        item
        for item in payload["items"]
        if item["ma_nganh"] == "7220104"
        and item["admission_year"] == 2026
        and item["field"] == "ten_nganh"
        and item["record_type"] == record_type
    ]
    assert len(items) == 1, f"expected one {record_type}/7220104/2026/ten_nganh item"
    return items[0]


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def app(store_root: Path, monkeypatch) -> FastAPI:
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    # The real startup wiring, over a throwaway root — the same call app
    # startup makes, so the ledger the assertions read is the ledger the
    # routes append to.
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ledger(store_root: Path) -> AuditLedger:
    """The same file the app's ledger writes, opened independently."""
    return AuditLedger(store_root / "audit" / "ledger.jsonl")


@pytest.fixture
def disk(store_root: Path) -> CanonicalRecordStore:
    """A second, independent store over the same root — reads real bytes."""
    return CanonicalRecordStore(store_root)


# -- criteria 1 + 9: the queue over HTTP ---------------------------------------


def test_conflicts_endpoint_lists_seed_pair(client: TestClient) -> None:
    """GET /admin/conflicts answers with the seed report's conflicts; the
    criterion-1 pair carries exactly the two distinct values, statements keyed
    by source document. Criterion 9's accounting is in the same response."""
    response = client.get("/admin/conflicts", headers=_headers("viewer"))
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == 74
    assert payload["seed_reported_items"] == 74
    assert payload["deduped"] == 0
    assert payload["dedup_reason"] is None
    assert payload["open"] == 74
    assert payload["resolved"] == 0

    item = _pair_item(payload)
    assert set(item["statements"].values()) == {"Hán Nôm", "Hán - Nôm"}
    assert len(item["statements"]) == 3
    assert item["status"] == "open"
    assert item["origin"] == "seed_report"


# -- criterion 3: the reason is mandatory --------------------------------------


def test_resolve_without_reason_is_422_and_changes_nothing(
    client: TestClient, ledger: AuditLedger, disk: CanonicalRecordStore
) -> None:
    """Blank and whitespace-only reasons are refused at the route, no event is
    appended, and the item stays open. The winner record is seeded so a
    mutation that drops the route's check cannot hide behind an unrelated 404:
    the request then reaches the service and the audit layer's
    ``REASON_REQUIRED`` raises — the ``ValueError`` escapes the route and
    breaks the exact-422 assert (measured: ``ValueError: 'conflict_override'
    requires a reason`` at ``audit.py:199``), with the no-event and still-open
    asserts proving the second guard held."""
    _seed_winner_record(disk)
    listing = client.get("/admin/conflicts", headers=_headers("viewer"))
    assert listing.status_code == 200
    conflict_id = _pair_item(listing.json())["conflict_id"]

    for blank in ("", "     "):
        response = client.post(
            f"/admin/conflicts/{conflict_id}/resolve",
            json={"winner_source_document_id": WINNER_DOC, "reason": blank},
            headers=_headers("admin"),
        )
        assert response.status_code == 422, (blank, response.text)
    assert ledger.find(action=AuditAction.CONFLICT_OVERRIDE) == ()

    after = client.get("/admin/conflicts", headers=_headers("viewer"))
    assert _pair_item(after.json())["status"] == "open"


# -- criterion 4 (+ 5, 6-actor): a real decision, properly audited --------------


def _seed_winner_record(disk: CanonicalRecordStore) -> str:
    """A published version carrying the winning statement, so the resolve can
    point its event at a real record version."""
    record_id = "NGANH:kind:7220104:2026:tt202611"
    disk.write_record(
        CanonicalRecord(
            record_id=record_id,
            record_type=RecordType.NGANH,
            version=1,
            admission_year=2026,
            effective_from=date(2026, 1, 1),
            status=RecordStatus.PUBLISHED,
            authority_level=AuthorityLevel.HUSC_OFFICIAL,
            source_document_id=WINNER_DOC,
            values={"ma_nganh": "7220104", "ten_nganh": "Hán - Nôm"},
            text="ten_nganh=Hán - Nôm",
        )
    )
    return record_id


def test_resolve_emits_conflict_override_event(
    client: TestClient, ledger: AuditLedger, disk: CanonicalRecordStore
) -> None:
    """The full flow: resolve -> one event with the credential's actor (not
    the body's claim), target = the winning version, trimmed reason, chain
    intact; item flips to resolved; a second resolve answers 409 and adds
    nothing."""
    record_id = _seed_winner_record(disk)
    listing = client.get("/admin/conflicts", headers=_headers("viewer"))
    conflict_id = _pair_item(listing.json())["conflict_id"]

    response = client.post(
        f"/admin/conflicts/{conflict_id}/resolve",
        json={
            "winner_source_document_id": WINNER_DOC,
            "reason": "  quyêt định cuối cùng, căn cứ công văn 345  ",
            # A body claiming a different actor must be ignored: the event's
            # actor comes from the credential.
            "actor_id": "gian-dinh",
        },
        headers=_headers("admin"),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "resolved"
    assert body["resolution"]["winner_source_document_id"] == WINNER_DOC
    assert body["resolution"]["reason"] == "quyêt định cuối cùng, căn cứ công văn 345"

    events = ledger.find(action=AuditAction.CONFLICT_OVERRIDE)
    assert len(events) == 1, "exactly one event per decision"
    event = events[0].event
    assert event.actor_id == "cb.quan-tri"  # the credential, not the body
    assert event.role == "admin"
    assert event.target == record_id
    assert event.after_version == f"{record_id}@1"
    assert event.reason == "quyêt định cuối cùng, căn cứ công văn 345"
    assert event.review_id == conflict_id
    assert ledger.verify().ok

    second = client.post(
        f"/admin/conflicts/{conflict_id}/resolve",
        json={"winner_source_document_id": WINNER_DOC, "reason": "quyêt định khác"},
        headers=_headers("admin"),
    )
    assert second.status_code == 409, second.text
    assert len(ledger.find(action=AuditAction.CONFLICT_OVERRIDE)) == 1

    after = client.get("/admin/conflicts", headers=_headers("viewer"))
    assert _pair_item(after.json())["status"] == "resolved"
    assert after.json()["resolved"] == 1


# -- criterion 6: permission fail-closed ----------------------------------------


def test_resolve_requires_the_override_grant(
    client: TestClient, disk: CanonicalRecordStore
) -> None:
    """A viewer may read the queue but not decide it (403); an unauthenticated
    caller is refused before RBAC even speaks (422 from the required header).
    The winner record is seeded so the mutation is honest: remove the resolve
    route's ``require_action`` -> the viewer's resolve *succeeds* (200) and
    this test goes RED on the 403 assert, not on an unrelated 404."""
    _seed_winner_record(disk)
    listing = client.get("/admin/conflicts", headers=_headers("viewer"))
    conflict_id = _pair_item(listing.json())["conflict_id"]
    resolve = {"winner_source_document_id": WINNER_DOC, "reason": "quyêt định"}

    denied = client.post(
        f"/admin/conflicts/{conflict_id}/resolve", json=resolve, headers=_headers("viewer")
    )
    assert denied.status_code == 403, denied.text

    anonymous = client.post(f"/admin/conflicts/{conflict_id}/resolve", json=resolve)
    assert anonymous.status_code == 422, anonymous.text


def test_resolve_refuses_unknown_item_and_foreign_winner(
    client: TestClient, disk: CanonicalRecordStore
) -> None:
    """The 404/422 refusals at the HTTP layer: no such queued conflict, and a
    winner that states nothing in this conflict. Both leave the item open."""
    _seed_winner_record(disk)
    listing = client.get("/admin/conflicts", headers=_headers("viewer"))
    conflict_id = _pair_item(listing.json())["conflict_id"]

    missing = client.post(
        "/admin/conflicts/0000000000000000/resolve",
        json={"winner_source_document_id": WINNER_DOC, "reason": "quyêt định"},
        headers=_headers("admin"),
    )
    assert missing.status_code == 404, missing.text

    foreign = client.post(
        f"/admin/conflicts/{conflict_id}/resolve",
        json={"winner_source_document_id": "doc-that-states-nothing", "reason": "quyêt định"},
        headers=_headers("admin"),
    )
    assert foreign.status_code == 422, foreign.text

    after = client.get("/admin/conflicts", headers=_headers("viewer"))
    assert _pair_item(after.json())["status"] == "open"
