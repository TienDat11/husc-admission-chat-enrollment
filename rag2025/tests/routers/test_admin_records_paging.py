"""GET /admin/records paging and ngành matching, without a live store.

Covers the defaults (no limit/offset = every record, as before), slicing with
total = matches before paging, a blank ngành filtering nothing, folded
ngành matching, and the blocking read running off the event loop.
"""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from src.routers import admin


def _rec(record_id, year, *names):
    return SimpleNamespace(record_id=record_id, admission_year=year, status=None, names=names)


@pytest.fixture
def fake_store(monkeypatch):
    records = [
        _rec("r-2025-b", 2025, "Công nghệ thông tin"),
        _rec("r-2026-a", 2026, "Hán Nôm"),
        _rec("r-2026-b", 2026, "Công nghệ thông tin"),
        _rec("r-2024-z", 2024, "Sư phạm Toán"),
    ]
    seen_threads = []

    def latest(_store):
        seen_threads.append(threading.get_ident())
        return list(records)

    monkeypatch.setattr(admin, "_lifecycle_service", lambda request: SimpleNamespace(store=object()))
    monkeypatch.setattr(admin, "_latest_versions", latest)
    monkeypatch.setattr(admin, "_nganh_candidates", lambda record: record.names)
    # Return the (records, total) pair so the assertions read plain tuples.
    monkeypatch.setattr(admin, "RecordListResponse", lambda records, total: (records, total))
    return seen_threads


def _list(**overrides):
    params = dict(nganh=None, year=None, status=None, limit=None, offset=0)
    params.update(overrides)
    return asyncio.run(admin.list_records(request=None, **params))


def _ids(records):
    return [record.record_id for record in records]


def test_defaults_return_every_record_newest_year_first(fake_store):
    records, total = _list()
    assert total == 4
    assert _ids(records) == ["r-2026-a", "r-2026-b", "r-2025-b", "r-2024-z"]


def test_limit_and_offset_slice_after_sorting_and_total_counts_matches(fake_store):
    records, total = _list(limit=2, offset=1)
    assert total == 4
    assert _ids(records) == ["r-2026-b", "r-2025-b"]


def test_offset_past_the_end_is_empty_with_total_intact(fake_store):
    records, total = _list(offset=10)
    assert records == [] and total == 4


def test_blank_nganh_filters_nothing_and_folded_nganh_matches(fake_store):
    assert _list(nganh="   ")[1] == 4
    records, total = _list(nganh="han nom")
    assert total == 1 and _ids(records) == ["r-2026-a"]
    records, total = _list(nganh="CÔNG NGHỆ")
    assert total == 2 and _ids(records) == ["r-2026-b", "r-2025-b"]


def test_blocking_read_runs_off_the_event_loop_thread(fake_store):
    main_thread = threading.get_ident()
    _list()
    assert fake_store and all(tid != main_thread for tid in fake_store)
