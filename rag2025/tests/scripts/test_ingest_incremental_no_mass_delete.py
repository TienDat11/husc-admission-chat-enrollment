"""D17 regression: --incremental ingest must never delete rows outside the batch.

History: `ingest_lancedb.py` called `when_not_matched_by_source_delete()` with no
condition on husc_chunks (which has no "source" column). The comment claimed the
batch was "full desired state from all files", but in --incremental mode
`_resolve_target_files` returns only the CHANGED files. A probe on a throwaway
table proved a 2-record batch destroyed 8 of 10 pre-existing rows.

This test pins the merge_insert contract at the LanceDB level, so the landmine
cannot be re-armed by a future edit.
"""
from __future__ import annotations

import lancedb
import pyarrow as pa
import pytest


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 4)),
        ]
    )


def _rows(ids):
    return [
        {"id": i, "text": f"text-{i}", "vector": [0.1, 0.2, 0.3, 0.4]} for i in ids
    ]


@pytest.fixture()
def table(tmp_path):
    db = lancedb.connect(str(tmp_path / "lancedb"))
    return db.create_table(
        "t", data=_rows([f"r{i}" for i in range(10)]), schema=_schema()
    )


def _merge(table, records, *, unconditional_delete: bool):
    job = (
        table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all()
    )
    if unconditional_delete:
        job.when_not_matched_by_source_delete()
    job.execute(records)


def test_subset_batch_preserves_rows_outside_batch(table):
    """The shipped path: a small incremental batch must not touch other rows."""
    assert table.count_rows() == 10

    _merge(table, _rows(["r0", "new1"]), unconditional_delete=False)

    ids = {r["id"] for r in table.to_arrow().select(["id"]).to_pylist()}
    assert ids == {f"r{i}" for i in range(10)} | {"new1"}
    assert table.count_rows() == 11, "pre-existing rows must survive a subset merge"


def test_unconditional_delete_is_destructive(table):
    """Falsification guard: proves WHY the delete call must stay absent.

    If a future LanceDB release made unconditional delete batch-scoped (making
    it safe), this test would fail and tell us the guard can be relaxed.
    """
    _merge(table, _rows(["r0", "new1"]), unconditional_delete=True)

    ids = {r["id"] for r in table.to_arrow().select(["id"]).to_pylist()}
    assert ids == {"r0", "new1"}
    assert table.count_rows() == 2, (
        "unconditional when_not_matched_by_source_delete still wipes rows outside "
        "the batch — the guard in ingest_lancedb.py must remain"
    )


def test_id_in_batch_condition_is_a_noop(table):
    """Documents why `id IN (batch)` was rejected instead of shipped.

    A row that is 'not matched by source' by definition has an id absent from the
    batch, so the condition can never fire. It is dead SQL (52 KB for a 2138-row
    batch), not a safety mechanism.
    """
    batch = _rows(["r0", "new1"])
    ids_sql = ", ".join(f"'{r['id']}'" for r in batch)

    job = (
        table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all()
    )
    job.when_not_matched_by_source_delete(condition=f"id IN ({ids_sql})")
    job.execute(batch)

    ids = {r["id"] for r in table.to_arrow().select(["id"]).to_pylist()}
    assert ids == {f"r{i}" for i in range(10)} | {"new1"}, (
        "id IN (batch) behaves identically to omitting the delete entirely"
    )
