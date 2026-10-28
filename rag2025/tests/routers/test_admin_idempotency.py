"""Idempotency of the mutating admin routes — ticket 27.

The ticket asks for replay protection on publish and rollback. Both are already
guarded against *state* replay (the lifecycle state machine has no self-loop for
publish; an explicit-target rollback hits ``TargetIsActiveError``). This module
covers what those guards cannot: the **implicit-target rollback**, whose target
is resolved from the current pointer, so a replay resolves a *different* target
and walks back a second release.

The tests are ordered by what they prove:

1. the hazard is real — without a key, a replayed implicit rollback moves two
   releases (this is the bug the ticket names, reproduced);
2. with a key, the replay returns the first response and does not re-run;
3. the key is per actor, and per intention (same key + different body → 422);
4. a failed attempt frees its key, so the operator's retry works;
5. a deployment with no store refuses a keyed request rather than silently
   running it twice.

Isolation: one throwaway store root and LanceDB table per test, no network, no
LLM — the retrieval half is a real LanceDB table, as in the ticket-22 suite.
"""
from __future__ import annotations

import importlib.util
import json
import sys
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
from src.domain.release import ReleaseManifest  # noqa: E402
from src.infrastructure.canonical_store import (  # noqa: E402
    publish_pointer,
    write_manifest,
)
from src.infrastructure.lancedb_versioning import tag_release  # noqa: E402
from src.routers.idempotency_guard import (  # noqa: E402
    IDEMPOTENCY_HEADER,
    REPLAY_HEADER,
)
from src.services.admin_rollback import release_tag  # noqa: E402
from src.services.idempotency import (  # noqa: E402
    IdempotencyStore,
    request_fingerprint,
)
from src.services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    LanceDBRetrieverConfig,
)

MAIN_PATH = ROOT / "src" / "main.py"
TABLE_NAME = "husc_chunks"

ACTORS = {
    "admin": ("cb.quan-tri", "admin"),
    "admin2": ("cb.quan-tri-2", "admin"),
}
# ``admin2`` is a second *actor* holding the same role, not a second role: the
# registry validates the role against the matrix, so "admin2" is not a role it
# would accept.
TOKENS = {"admin": "tok-admin", "admin2": "tok-admin-2"}
OPS_TOKEN = "shared-ops-token"

# A three-release lineage: R3 active, R2 behind it, R1 behind that. Two hops is
# what makes the replay hazard visible — one replay has to move a second release.
R1_ID = "R-2026-07-01"
R2_ID = "R-2026-08-01"
R3_ID = "R-2026-09-01"

REASON = "khôi phục sau sự cố dữ liệu 2026"


def _headers(role: str = "admin", key: str | None = None) -> dict[str, str]:
    headers = {"X-Admin-Token": TOKENS[role]}
    if key is not None:
        headers[IDEMPOTENCY_HEADER] = key
    return headers


def _row(row_id: str, text: str, release_id: str) -> dict:
    return {
        "id": row_id,
        "text": text,
        "source": "de-an-tuyen-sinh-2026.pdf",
        "chunk_id": row_id,
        "metadata_json": json.dumps({"release_id": release_id}, ensure_ascii=False),
        "release_id": release_id,
        "vector": [0.1, 0.2, 0.3, 0.4],
    }


def _releases_dir(store_root: Path) -> Path:
    return store_root / "releases"


def _pointer_path(store_root: Path) -> Path:
    return _releases_dir(store_root) / "active_release.json"


def _active(store_root: Path) -> str:
    return json.loads(_pointer_path(store_root).read_text(encoding="utf-8"))["release_id"]


def _seed_release(store_root: Path, release_id: str, *, parent: str | None) -> None:
    write_manifest(
        _releases_dir(store_root) / release_id,
        ReleaseManifest(
            release_id=release_id,
            parent_release_id=parent,
            lancedb_tag=release_tag(release_id),
        ),
    )


@pytest.fixture(scope="module")
def real_main():
    """The real app, as ``src/main.py`` builds it — the seam accessors included."""
    spec = importlib.util.spec_from_file_location("husc_idem_main", MAIN_PATH)
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
                {"token": TOKENS[key], "actor_id": actor, "role": role}
                for key, (actor, role) in ACTORS.items()
            ]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)
    configure_canonical_record_store(
        real_main.app, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    return real_main.app


@pytest.fixture
def client(app) -> TestClient:
    """No lifespan: this route needs routers and wiring, not the models."""
    return TestClient(app)


@pytest.fixture
def lineage(real_main, monkeypatch, tmp_path: Path, store_root: Path):
    """R3 active, R2 and R1 behind it, all tagged on one real table."""
    db = lancedb.connect(str(tmp_path / "lancedb"))
    table = db.create_table(TABLE_NAME, data=[_row("r1-a", "Điểm chuẩn 2026", R1_ID)])
    tag_release(table, R1_ID)
    table.add([_row("r2-a", "Điểm chuẩn 2026 ngành CNTT là 25,5", R2_ID)])
    tag_release(table, R2_ID)
    table.add([_row("r3-a", "Điểm chuẩn 2026 ngành KTPM là 24,0", R3_ID)])
    tag_release(table, R3_ID)

    retriever = LanceDBRetriever(
        LanceDBRetrieverConfig(
            uri=str(tmp_path / "lancedb"), table_name=TABLE_NAME, embedding_dim=4
        )
    )
    monkeypatch.setattr(real_main, "lancedb_retriever_service", retriever)
    monkeypatch.setattr(real_main, "hybrid_search_service", None)

    _seed_release(store_root, R1_ID, parent=None)
    _seed_release(store_root, R2_ID, parent=R1_ID)
    _seed_release(store_root, R3_ID, parent=R2_ID)
    publish_pointer(_pointer_path(store_root), R3_ID)
    return retriever


def _post_rollback(client: TestClient, *, key: str | None = None, role: str = "admin",
                   reason: str = REASON, target: str | None = None):
    body: dict = {"reason": reason}
    if target is not None:
        body["target_release_id"] = target
    return client.post(
        "/admin/rollback", json=body, headers=_headers(role, key)
    )


# ---------------------------------------------------------------------------
# 1. the hazard, reproduced — this is the bug ticket 27 names
# ---------------------------------------------------------------------------


def test_without_a_key_a_replayed_implicit_rollback_moves_two_releases(
    client: TestClient, store_root: Path, lineage
):
    """The failure mode: a retry of an implicit rollback walks back one hop too far.

    No ``Idempotency-Key``, so this is the pre-ticket behaviour — reproduced
    deliberately, because a ticket that claims a hazard must show it. R3→R2,
    then the retry resolves R2→R1 and succeeds again. Both calls return 200 and
    report a different release, so the caller cannot tell the second was a
    duplicate of the first.
    """
    assert _active(store_root) == R3_ID

    first = _post_rollback(client)
    second = _post_rollback(client)

    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["release_id"] == R2_ID
    assert second.json()["release_id"] == R1_ID
    # Two releases behind the original pointer: the duplicate did real damage.
    assert _active(store_root) == R1_ID


def test_an_explicit_target_replay_is_already_refused_without_a_key(
    client: TestClient, store_root: Path, lineage
):
    """The contrast: an explicit target cannot be replayed — state catches it.

    This is why the ticket's remaining work is the *implicit* case. Naming the
    target makes the second call find it already active, so it is refused rather
    than applied twice.
    """
    assert _post_rollback(client, target=R2_ID).status_code == 200

    replay = _post_rollback(client, target=R2_ID)

    assert replay.status_code == 409
    assert _active(store_root) == R2_ID


# ---------------------------------------------------------------------------
# 2. with a key, a replay is absorbed
# ---------------------------------------------------------------------------


def test_replayed_rollback_returns_the_first_response_and_moves_nothing(
    client: TestClient, store_root: Path, lineage
):
    """Same key, same body → the first response, and the pointer does not move again."""
    first = _post_rollback(client, key="rb-1")
    assert first.status_code == 200
    assert first.json()["release_id"] == R2_ID
    assert REPLAY_HEADER not in first.headers  # a fresh run, not a replay

    replay = _post_rollback(client, key="rb-1")

    assert replay.status_code == 200
    assert replay.headers[REPLAY_HEADER] == "true"
    assert replay.json() == first.json()
    # The proof it did not re-run: one hop back, not two.
    assert _active(store_root) == R2_ID


def test_a_replay_emits_no_second_audit_event(client: TestClient, store_root: Path, lineage):
    """The ledger records one rollback, not two.

    The strongest available statement that the operation did not re-run: a
    second execution would append a second ROLLBACK event.
    """
    from src.domain.audit import AuditAction
    from src.infrastructure.audit_ledger import AuditLedger

    def rollback_events():
        ledger = AuditLedger(store_root / "audit" / "ledger.jsonl")
        return ledger.find(action=AuditAction.ROLLBACK)

    _post_rollback(client, key="rb-ledger")
    after_first = len(rollback_events())

    _post_rollback(client, key="rb-ledger")

    assert len(rollback_events()) == after_first


def test_a_new_key_runs_a_real_second_rollback(client: TestClient, store_root: Path, lineage):
    """A different key is a different request — the guard must not over-absorb.

    The inverse of the replay test: absorbing every repeated body would break
    the legitimate case of rolling back twice on purpose.
    """
    assert _post_rollback(client, key="rb-a").json()["release_id"] == R2_ID
    second = _post_rollback(client, key="rb-b")

    assert second.status_code == 200
    assert REPLAY_HEADER not in second.headers
    assert second.json()["release_id"] == R1_ID


# ---------------------------------------------------------------------------
# 3. the key is scoped — per actor, per intention
# ---------------------------------------------------------------------------


def test_the_same_key_from_another_actor_is_a_different_request(
    client: TestClient, store_root: Path, lineage
):
    """Keys are per actor: one operator's key cannot absorb another's request.

    If keys were global, two operators both numbering their retries "1" would
    have the second one's rollback silently swallowed.
    """
    assert _post_rollback(client, key="retry-1", role="admin").json()["release_id"] == R2_ID

    other = _post_rollback(client, key="retry-1", role="admin2")

    assert other.status_code == 200
    assert REPLAY_HEADER not in other.headers
    assert other.json()["release_id"] == R1_ID


def test_the_same_key_with_a_different_body_is_refused(
    client: TestClient, store_root: Path, lineage
):
    """Reusing a key for a different intention is a client bug, refused with 422.

    Answering it from the stored record would report the first rollback's result
    for a request that asked for something else.
    """
    assert _post_rollback(client, key="rb-2").status_code == 200

    different = _post_rollback(client, key="rb-2", reason="lý do khác hoàn toàn")

    assert different.status_code == 422
    assert "different request" in different.json()["detail"]
    assert _active(store_root) == R2_ID


# ---------------------------------------------------------------------------
# 4. failure frees the key; success does not
# ---------------------------------------------------------------------------


def test_a_failed_attempt_frees_its_key_for_the_retry(
    client: TestClient, store_root: Path, lineage
):
    """A 5xx must not consume the key — the retry that fixes it must be allowed.

    Simulated with a target that does not exist (404): the operator corrects the
    body and retries under the same key, which is the reason they sent one.
    """
    failed = _post_rollback(client, key="rb-3", target="R-khong-ton-tai")
    assert failed.status_code == 404

    retry = _post_rollback(client, key="rb-3", target=R2_ID)

    assert retry.status_code == 200, retry.text
    assert retry.json()["release_id"] == R2_ID


def test_a_completed_key_is_not_reusable_for_a_second_effect(
    client: TestClient, store_root: Path, lineage
):
    """A successful key stays consumed: the retry is absorbed, not re-run."""
    _post_rollback(client, key="rb-4")
    absorbed = _post_rollback(client, key="rb-4")

    assert absorbed.status_code == 200
    assert absorbed.headers[REPLAY_HEADER] == "true"
    assert _active(store_root) == R2_ID


# ---------------------------------------------------------------------------
# 5. an unconfigured deployment refuses rather than double-runs
# ---------------------------------------------------------------------------


def test_a_keyed_request_without_a_store_is_refused(
    client: TestClient, app, store_root: Path, lineage
):
    """No store configured + a key sent → 503, never a silent second run.

    A caller who asked for a replay guarantee must not be told they got one by a
    deployment that cannot provide it.
    """
    del app.state.idempotency_store

    refused = _post_rollback(client, key="rb-5")

    assert refused.status_code == 503
    assert "idempotency_store" in refused.json()["detail"]
    # And nothing moved.
    assert _active(store_root) == R3_ID


def test_a_request_without_a_key_is_unaffected_by_the_guard(
    client: TestClient, app, store_root: Path, lineage
):
    """The header is optional: existing callers keep working with no store."""
    del app.state.idempotency_store

    assert _post_rollback(client).status_code == 200


# ---------------------------------------------------------------------------
# 6. the store's own contract
# ---------------------------------------------------------------------------


def test_fingerprint_ignores_field_order_but_not_values():
    """The fingerprint is over canonical JSON: key order is not a difference."""
    assert request_fingerprint({"a": 1, "b": 2}) == request_fingerprint({"b": 2, "a": 1})
    assert request_fingerprint({"a": 1}) != request_fingerprint({"a": 2})


def test_a_key_is_refused_when_blank_or_overlong(tmp_path: Path):
    """Blank and overlong keys are refused, not truncated onto a shared bucket."""
    from src.services.idempotency import IdempotencyKeyInvalid

    store = IdempotencyStore(tmp_path)
    for bad in ("", "   ", "x" * 201):
        with pytest.raises(IdempotencyKeyInvalid):
            store.begin(actor_id="cb.a", key=bad, operation="op", body={})


def test_an_in_flight_claim_refuses_a_concurrent_duplicate(tmp_path: Path):
    """A second request under a live claim gets a conflict, not a second run."""
    from src.services.idempotency import IdempotencyInProgress

    store = IdempotencyStore(tmp_path)
    assert store.begin(actor_id="cb.a", key="k", operation="op", body={}) is None

    with pytest.raises(IdempotencyInProgress):
        store.begin(actor_id="cb.a", key="k", operation="op", body={})


def test_two_actors_with_the_same_key_do_not_collide(tmp_path: Path):
    """The record path is keyed by actor *and* key, so tenants cannot collide."""
    store = IdempotencyStore(tmp_path)
    store.begin(actor_id="cb.a", key="1", operation="op", body={})
    store.complete(actor_id="cb.a", key="1", status_code=200, response={"who": "a"})

    # A different actor with the same key owns a separate claim.
    assert store.begin(actor_id="cb.b", key="1", operation="op", body={}) is None
