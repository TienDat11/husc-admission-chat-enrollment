"""Tests for the T16 serve probe (K3) + queue out_of_scope flag (K4).

AC pin:
- probe append -> stamp -> durations flow (real minutes, never 0-filled);
- empty log -> no durations;
- out_of_scope flag round-trips through queue file (old rows default False).
"""
from __future__ import annotations

from src.services.serve_probe import (
    append_probe_entry,
    k3_durations_minutes,
    probe_log_path,
    read_probe_entries,
    stamp_served,
    stamp_served_from_chunks,
)
from src.services.unanswered_queue import UnansweredQueueService


def test_probe_append_stamp_durations(tmp_path):
    append_probe_entry(
        data_dir=tmp_path, release_id="r1",
        upload_received_at="2026-10-08T10:00:00+00:00",
        upload_received_at_known=True,
        published_at="2026-10-08T10:05:00+00:00",
    )
    assert probe_log_path(tmp_path).is_file()
    assert stamp_served(data_dir=tmp_path, release_id="r1",
                        served_at="2026-10-08T10:30:00+00:00") is True
    entries = read_probe_entries(probe_log_path(tmp_path))
    durations = k3_durations_minutes(entries)
    assert durations == [("r1", 30.0)]
    # double stamp writes nothing
    assert stamp_served(data_dir=tmp_path, release_id="r1") is False


def test_probe_stamp_unknown_release_writes_nothing(tmp_path):
    assert stamp_served(data_dir=tmp_path, release_id="nope") is False
    assert not probe_log_path(tmp_path).is_file()


def test_probe_stamp_from_chunks_matches_release(tmp_path):
    append_probe_entry(
        data_dir=tmp_path, release_id="hook-r9", chunk_release_id="chunk-r9",
        upload_received_at="2026-10-08T10:00:00+00:00",
        upload_received_at_known=True,
    )
    chunks = [{"metadata": {"release_id": "chunk-r9"}, "chunk_id": "c1"}]
    assert stamp_served_from_chunks(tmp_path, chunks) == 1
    entries = read_probe_entries(probe_log_path(tmp_path))
    assert entries["hook-r9"]["served_at"] is not None
    # served entries with unknown upload are skipped for durations
    append_probe_entry(data_dir=tmp_path, release_id="r2")
    assert len(k3_durations_minutes(entries)) == 1


class _RetrievedDocumentLike:
    """Duck-typed stand-in for lancedb_retrieval.RetrievedDocument
    (``to_dict`` + ``metadata`` attr) — the real class drags in the
    LanceDB adapter chain, which this unit must not import."""

    def __init__(self, release_id: str, record_id: str | None = None) -> None:
        self.metadata = {"release_id": release_id}
        if record_id:
            self.metadata["record_id"] = record_id
        self.chunk_id = "c-build-1"

    def to_dict(self):
        return {"chunk_id": self.chunk_id, "metadata": dict(self.metadata)}


def test_probe_requires_record_match_same_release(tmp_path):
    """Spec 02 §9: the probe closes only when a chunk OF THE PROBE'S
    RECORD comes back. Another record of the same release must NOT
    close it — same release, wrong record."""
    append_probe_entry(
        data_dir=tmp_path, release_id="rel-1", chunk_release_id="rel-1",
        record_id="rec-A",
        upload_received_at="2026-10-08T10:00:00+00:00",
        upload_received_at_known=True,
    )
    wrong_record = [_RetrievedDocumentLike("rel-1", "rec-B")]
    assert stamp_served_from_chunks(tmp_path, wrong_record) == 0
    entries = read_probe_entries(probe_log_path(tmp_path))
    assert entries["rel-1"]["served_at"] is None
    # same release, RIGHT record -> closes
    right_record = [_RetrievedDocumentLike("rel-1", "rec-A")]
    assert stamp_served_from_chunks(tmp_path, right_record) == 1
    entries = read_probe_entries(probe_log_path(tmp_path))
    assert entries["rel-1"]["served_at"] is not None


def test_probe_closes_on_build_id_when_hook_id_differs(tmp_path):
    """The join that makes K3 real: the hook mints its publish-time id
    AFTER chunks were built, so served chunks carry the older build-time
    id. The stamp must close the probe via ``chunk_release_id``."""
    append_probe_entry(
        data_dir=tmp_path, release_id="hook-20261009-0001",
        chunk_release_id="build-20261008-0007",
        upload_received_at="2026-10-08T10:00:00+00:00",
        upload_received_at_known=True,
    )
    served = [_RetrievedDocumentLike("build-20261008-0007")]
    assert stamp_served_from_chunks(tmp_path, served) == 1
    entries = read_probe_entries(probe_log_path(tmp_path))
    assert entries["hook-20261009-0001"]["served_at"] is not None
    assert k3_durations_minutes(entries)[0][0] == "hook-20261009-0001"


def test_probe_empty_log_has_no_durations(tmp_path):
    assert read_probe_entries(tmp_path / "missing.jsonl") == {}
    assert k3_durations_minutes({}) == []


def test_out_of_scope_flag_round_trip(tmp_path):
    svc = UnansweredQueueService(data_dir=tmp_path)
    item = svc.push("cau hoi", "t1", "v2", "X")
    assert item.out_of_scope is False
    flagged = svc.mark_out_of_scope("t1")
    assert flagged.out_of_scope is True
    assert svc.items()[0].out_of_scope is True
    # unflag restores denominator membership
    assert svc.mark_out_of_scope("t1", False).out_of_scope is False


def test_out_of_scope_defaults_false_for_old_rows(tmp_path):
    svc = UnansweredQueueService(data_dir=tmp_path)
    svc.push("cau hoi", "t1", "v2", "X")
    import json

    rows = [
        json.loads(line) for line in
        (tmp_path / "unanswered" / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    del rows[0]["out_of_scope"]  # simulate a pre-flag row
    (tmp_path / "unanswered" / "queue.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")
    assert svc.items()[0].out_of_scope is False
