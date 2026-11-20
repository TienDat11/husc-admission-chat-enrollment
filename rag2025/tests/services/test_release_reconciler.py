"""ReleaseReconciler reads the active release and its recommender snapshot row."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.services import major_recommender
from src.services.release_reconciler import ReleaseReconciler
from src.services.release_snapshot import ReleaseSnapshotLoadError

ENTRIES = [
    {
        "major_code": "7480201",
        "method": "thi_thpt",
        "year": 2025,
        "diem_chuan": 24.5,
        "to_hop": ["A00"],
    }
]


class _Result:
    def __init__(self, row):
        self._row = row

    def mappings(self):
        return self

    def first(self):
        return self._row


class _Conn:
    def __init__(self, rows):
        self._rows = rows

    async def execute(self, stmt, params=None):
        table = "releases" if "FROM releases" in str(stmt) else "recommender_snapshots"
        return _Result(self._rows.get(table))


class _ConnectCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _AsyncEngine:
    """Stands in for SQLAlchemy AsyncEngine: connect() is an async context manager."""

    def __init__(self, rows):
        self._conn = _Conn(rows)

    def connect(self):
        return _ConnectCtx(self._conn)


def _app(rows):
    return SimpleNamespace(state=SimpleNamespace(db_engine=_AsyncEngine(rows)))


@pytest.fixture(autouse=True)
def _reset_recommender_snapshot():
    yield
    major_recommender.set_snapshot(None)


def test_tick_reads_active_release_through_async_engine():
    rows = {"releases": {"release_id": "REL29B", "corpus_stamp": []}}
    seen: list[str] = []

    async def load(release_id):
        seen.append(release_id)
        return SimpleNamespace(release_id=release_id)

    app = _app(rows)
    asyncio.run(ReleaseReconciler(app, load_snapshot=load)._tick())

    assert seen == ["REL29B"]
    assert app.state.release_snapshot.release_id == "REL29B"


def test_default_load_puts_snapshot_entries_where_recommender_reads_them():
    row = {
        "data_year": 2026,
        "diem_chuan_history": ENTRIES,
        "majors": [],
        "to_hop": {},
        "chi_tieu": [],
        "methods": ["thi_thpt"],
        "tuition": {},
        "priority_rules": {"KV_BONUS": {"KV1": "0.75"}},
    }
    snap = asyncio.run(
        ReleaseReconciler(_app({"recommender_snapshots": row}))._default_load("REL29B")
    )

    assert snap.recommender_snapshot["entries"] == ENTRIES
    assert snap.recommender_snapshot["data_year"] == 2026
    assert major_recommender._SNAPSHOT is snap


def test_default_load_without_snapshot_row_raises():
    with pytest.raises(ReleaseSnapshotLoadError) as exc:
        asyncio.run(ReleaseReconciler(_app({}))._default_load("REL29X"))

    assert exc.value.release_id == "REL29X"
