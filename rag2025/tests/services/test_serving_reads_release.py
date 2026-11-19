"""U8: serving paths read ReleaseSnapshot from RAM, never data/major_codes/*.json."""

from __future__ import annotations

import src.services.guardrail as gr
import src.services.major_code_validator as mcv
import src.services.major_recommender as mr
from src.services.major_recommender import recommend, set_snapshot, whatif_probability
from src.services.major_code_validator import (
    get_whitelist,
    set_snapshot_whitelist,
)
from src.services.guardrail import set_snapshot_allowed
from src.services.release_snapshot import ReleaseSnapshot


def _make_snapshot() -> ReleaseSnapshot:
    return ReleaseSnapshot(
        release_id="R-test",
        recommender_snapshot={
            "entries": [
                {
                    "major_code": "CNTT",
                    "major_name": "Cong nghe thong tin",
                    "method": "thi_thpt",
                    "year": 2025,
                    "diem_chuan": 24.0,
                    "to_hop": ["A", "A1"],
                },
                {
                    "major_code": "KHMT",
                    "major_name": "Khoa hoc may tinh",
                    "method": "thi_thpt",
                    "year": 2025,
                    "diem_chuan": 22.0,
                    "to_hop": ["A", "A1"],
                },
            ]
        },
        major_codes=frozenset({"CNTT", "KHMT"}),
        cutoff_scores={"CNTT": 24.0, "KHMT": 22.0},
        quota={"CNTT": 100, "KHMT": 80},
        tuition={"CNTT": 5000000.0, "KHMT": 4500000.0},
        priority_rules={},
        major_whitelist=frozenset({"CNTT", "KHMT"}),
        chunks_table="husc_chunks",
        corpus_stamp=[{"record_id": "r1", "version": 1}],
    )


def test_recommender_uses_snapshot_not_disk(monkeypatch):
    """Monkeypatch _HISTORY_PATH: serve path must not open data/major_codes/*.json."""
    class FakePath:
        def read_text(self, *a, **k):
            raise AssertionError("serve path must not call Path.read_text")

        def exists(self):
            return True

        def stat(self):
            class S:
                st_mtime = 0

            return S()

    monkeypatch.setattr(mr, "_HISTORY_PATH", FakePath())
    monkeypatch.setattr(mr, "_CACHE_BY_KEY", {})
    monkeypatch.setattr(mr, "_CACHE_LATEST", {})

    set_snapshot(_make_snapshot())
    recs = recommend(score=25.0, to_hop="A")
    assert len(recs) == 2
    codes = {r.major_code for r in recs}
    assert codes == {"CNTT", "KHMT"}


def test_whatif_uses_snapshot_not_disk(monkeypatch):
    monkeypatch.setattr(mr, "_HISTORY_PATH", object())
    monkeypatch.setattr(mr, "_CACHE_BY_KEY", {})
    monkeypatch.setattr(mr, "_CACHE_LATEST", {})

    set_snapshot(_make_snapshot())
    res = whatif_probability(score=25.0, major_code="CNTT")
    assert res.band != "unknown_major"
    assert res.latest_diem_chuan == 24.0


def test_whitelist_uses_snapshot(monkeypatch):
    monkeypatch.setattr(mcv, "_MAJOR_CODES_DIR", object())

    set_snapshot_whitelist(frozenset({"CNTT", "KHMT"}))
    wl = get_whitelist()
    assert wl == frozenset({"CNTT", "KHMT"})


def test_guardrail_uses_snapshot():
    set_snapshot_allowed(frozenset({"CNTT", "KHMT"}))
    assert gr._SNAPSHOT_ALLOWED == frozenset({"CNTT", "KHMT"})
