# M3 — Filter Pushdown Verdict

## Commands

```
D:/miniconda3/python.exe docs/evidence/probe4_filter_pushdown.py
```

lancedb 0.29.2, pyarrow 23.0.1

---

## PLAN_BEFORE_INDEX (no scalar indices)

```
ProjectionExec: expr=[id@7 as id, doc_id@8 as doc_id, vector@0 as vector, data_year@9 as data_year, status@1 as status, effective_from@2 as effective_from, effective_to@3 as effective_to, is_tombstone@4 as is_tombstone, _distance@6 as _distance]
  Take: columns="vector, status, effective_from, effective_to, is_tombstone, _rowid, _distance, (id), (doc_id), (data_year)"
    CoalesceBatchesExec: target_batch_size=1024
      GlobalLimitExec: skip=0, fetch=10
        FilterExec: _distance@6 IS NOT NULL
          SortExec: TopK(fetch=10), expr=[_distance@6 ASC NULLS LAST, _rowid@5 ASC NULLS LAST], preserve_partitioning=[false]
            KNNVectorDistance: metric=l2
              FilterExec: status@1 = current AND effective_from@2 <= 20260726 AND effective_to@3 >= 20260726 AND NOT is_tombstone@4
                LanceRead: uri=D:/tmp/tmpro5_01_1/probe.lance/data, projection=[vector, status, effective_from, effective_to, is_tombstone], num_fragments=1, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=status = Utf8("current") AND effective_from <= Int32(20260726) AND effective_to >= Int32(20260726) AND NOT is_tombstone, refine_filter=status = Utf8("current") AND effective_from <= Int32(20260726) AND effective_to >= Int32(20260726) AND NOT is_tombstone
```

Node names: `ProjectionExec` → `Take` → `CoalesceBatchesExec` → `GlobalLimitExec` → `FilterExec` (_distance) → `SortExec` → `KNNVectorDistance` → **`FilterExec`** (scalar predicate) → `LanceRead`

The scalar predicate `(status = 'current' AND effective_from <= ...)` appears as a **standalone `FilterExec` node** above `KNNVectorDistance`. The `full_filter` inside `LanceRead` shows the predicate is also pushed down to the scan level for pre-filtering, but there is still a separate `FilterExec` layer above it acting as a post-filter.

---

## PLAN_AFTER_INDEX (all 7 scalar indices created)

```
ProjectionExec: expr=[id@3 as id, doc_id@4 as doc_id, vector@0 as vector, data_year@5 as data_year, status@6 as status, effective_from@7 as effective_from, effective_to@8 as effective_to, is_tombstone@9 as is_tombstone, _distance@2 as _distance]
  Take: columns="vector, _rowid, _distance, (id), (doc_id), (data_year), (status), (effective_from), (effective_to), (is_tombstone)"
    CoalesceBatchesExec: target_batch_size=1024
      GlobalLimitExec: skip=0, fetch=10
        FilterExec: _distance@2 IS NOT NULL
          SortExec: TopK(fetch=10), expr=[_distance@2 ASC NULLS LAST, _rowid@1 ASC NULLS LAST], preserve_partitioning=[false]
            KNNVectorDistance: metric=l2
              LanceRead: uri=D:/tmp/tmpro5_01_1/probe.lance/data, projection=[vector], num_fragments=1, range_before=None, range_after=None, row_id=true, row_addr=false, full_filter=status = Utf8("current") AND effective_from <= Int32(20260726) AND effective_to >= Int32(20260726) AND NOT is_tombstone, refine_filter=--
                ScalarIndexQuery: query=AND(AND(AND([status = current]@status_idx,[effective_from <= 20260726]@effective_from_idx),[effective_to >= Int32(20260726)]@effective_to_idx),NOT([is_tombstone = true]@is_tombstone_idx))
```

Node names: `ProjectionExec` → `Take` → `CoalesceBatchesExec` → `GlobalLimitExec` → `FilterExec` (_distance) → `SortExec` → `KNNVectorDistance` → `LanceRead` → **`ScalarIndexQuery`**

Key observations:
- The standalone `FilterExec` (scalar predicate) that was present in PLAN_BEFORE_INDEX is **gone**.
- A new `ScalarIndexQuery` node appears at the bottom of the plan, directly below `LanceRead`.
- The predicate is fully resolved via `ScalarIndexQuery` using all 4 index-backed columns (`status_idx`, `effective_from_idx`, `effective_to_idx`, `is_tombstone_idx`).
- `LanceRead.full_filter` still contains the predicate string, but `refine_filter=--` (empty) indicates no post-scan refinement pass.

---

## PLAN_SCALAR_ONLY (no vector query, filter-only)

```
LanceRead: uri=D:/tmp/tmpro5_01_1/probe.lance/data, projection=[id, doc_id, vector, data_year, status, effective_from, effective_to, is_tombstone], num_fragments=1, range_before=None, range_after=Some(0..10), row_id=false, row_addr=false, full_filter=status = Utf8("current") AND effective_from <= Int32(20260726) AND effective_to >= Int32(20260726) AND NOT is_tombstone, refine_filter=--
  ScalarIndexQuery: query=AND(AND(AND([status = current]@status_idx,[effective_from <= Int32(20260726)]@effective_from_idx),[effective_to >= Int32(20260726)]@effective_to_idx),NOT([is_tombstone = true]@is_tombstone_idx))
```

Same pattern: `ScalarIndexQuery` resolves the predicate, no `FilterExec` wrapper.

---

## Node-name analysis

| Phase | Standalone FilterExec (scalar)? | ScalarIndexQuery? |
|-------|-------------------------------|-------------------|
| BEFORE_INDEX | YES — `FilterExec: status@1 = current AND ...` | NO |
| AFTER_INDEX | NO | YES — `ScalarIndexQuery: query=AND(...)` |
| SCALAR_ONLY | NO | YES — same pattern |

After indexing, the scalar predicate is **not** a detached `FilterExec`. It is absorbed into `ScalarIndexQuery` which sits below `LanceRead`. The predicate is pushed down and resolved via index lookups.

---

## BTREE vs BITMAP results

All 7 indices created successfully:

| Column | Type | Result |
|--------|------|--------|
| id | BTREE | OK |
| doc_id | BTREE | OK |
| data_year | BTREE | OK |
| effective_from | BTREE | OK |
| effective_to | BTREE | OK |
| status | BITMAP | OK |
| is_tombstone | BITMAP | OK |

No failures. Both BTREE and BITMAP accepted on all tested column types (string, int32, bool).

---

## Nested-vs-flat finding

Nested schema probe:
```
schema = [
  "id": string,
  "vector": list<float32, 8>,
  "metadata": struct<data_year: int32>
]
```

Results:
- `where("metadata.data_year = 2026")` → returned 5 rows (correct dotted-path filter works).
- `where("data_year = 2026")` → **RAISES**: `RuntimeError('lance error: LanceError(Schema): Schema error: No field named data_year. Valid fields are id, vector, metadata, _rowid, _rowaddr, _row_last_updated_at_version, _row_created_at_version, _rowoffset.')`

Dotted-path nested access works. Bare column name does not silently swallow — it raises a hard `RuntimeError`. This contradicts the plan's concern about silent swallow; LanceDB v0.29.2 properly rejects bare references to nested columns.

---

## B2 Blocker — temporal_filter.py line 40 verbatim

File: `rag2025/src/services/temporal_filter.py`

Line 40:
```python
        return f"data_year = {current_year}"
```

This emits a clause referencing the top-level column `data_year`. On a flat table with `data_year` as a direct field, this works fine. On a nested table where the column lives at `metadata.data_year`:

- The dotted form `metadata.data_year = 2026` would work (as proven by the NESTED_DOTTED_FILTER test returning 5 rows).
- The bare form `data_year = 2026` raises `RuntimeError` (Schema error: No field named data_year).

**Verdict on danger**: This is an error-raising failure, not a silent-swallow. The query crashes rather than returning wrong results. Still a blocker (B2) because it causes runtime failure for nested-schema tables, but less dangerous than silent incorrect results.

---

## Gate G3 verdict

Gate G3: "Predicate must not be a detached FilterExec after indexing."

Evidence: After all 7 scalar indices are created, PLAN_AFTER_INDEX shows zero standalone `FilterExec` nodes for the scalar predicate. The predicate is fully resolved via `ScalarIndexQuery`. The `LanceRead.full_filter` retains the predicate string (used for pre-scan pruning), but `refine_filter=--` confirms no post-scan re-evaluation is needed.

**G3: PASS**
