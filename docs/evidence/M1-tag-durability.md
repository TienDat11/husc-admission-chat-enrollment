# M1 Tag Durability Probe — Evidence Report

## Installed vs Pinned Versions

| | Pinned (requirements.txt) | Installed | Match? |
|---|---|---|---|
| lancedb | ==0.29.2 | 0.29.2 | YES |
| pyarrow | — | 23.0.1 | — |
| python | — | 3.13.2 (Anaconda MS v.1929 64-bit AMD64) | — |

**Finding: MAJOR — none.** Pinned matches installed.

## Command Outputs (verbatim)

### Command 1: Version check

```
lancedb 0.29.2
pyarrow 23.0.1
python 3.13.2 | packaged by Anaconda, Inc. | (MSC v.1929 64 bit (AMD64))
```

### Command 2: API signature check

```
has branches: False
restore sig: (self, version: 'Optional[Union[int, str]]' = None)
optimize sig: (self, *, cleanup_older_than: 'Optional[timedelta]' = None, delete_unverified: 'bool' = False, retrain: 'bool' = False)
```

### Command 3: Probe run output

```
version_after_tag = 1
version_after_add = 2
rows_latest = 2
optimize_ok = FAIL: RuntimeError('lance error: Cleanup error: 1 tagged version(s) have been marked for cleanup. Either set `error_if_tagged_old_versions=false` or delete the following tag(s) to enable cleanup: {"rel-A": 1}')
TAG_SURVIVES_OPTIMIZE = None
rows_at_tag_after_optimize = None
rows_back_to_latest = None
rows_after_restore = 1
RESTORE_BY_TAG_STRING = True
version_after_restore = 5
tags_update_ok = True
tags_list = {'rel-A': {'version': 1, 'manifest_size': 636}, 'prod': {'version': 2, 'manifest_size': 561}}
```

## Gate Results

| Gate | Expected | Actual | Result |
|------|----------|--------|--------|
| TAG_SURVIVES_OPTIMIZE | True | None (optimize itself failed) | **FAIL** |
| RESTORE_BY_TAG_STRING | True | True | **PASS** |
| tags.update availability | exists | True (succeeded) | **PASS** |
| hasattr(LanceTable, 'branches') | — | False | **INFO** |

## Key Findings

1. **optimize() blocks cleanup of tagged versions.** The sync wrapper `LanceTable.optimize()` does NOT expose `error_if_tagged_old_versions`. The inner async method accepts it, but the sync layer rejects it with `TypeError`. This means `cleanup_older_than` on a table with any tags will always raise `RuntimeError`.

2. **Tags survive by accident, not by design.** `TAG_SURVIVES_OPTIMIZE` is `None` because optimize never completes — it aborts before touching data. Tags are preserved only because the operation fails entirely.

3. **`restore("<tag string>")` works.** PASS. Passing a tag name string to `t.restore()` resolves to the correct version and restores the table to 1 row.

4. **`tags.update()` works.** Can update an existing tag to point at a newer version.

5. **No `LanceTable.branches` attribute.** Branch feature not available in v0.29.2.

## CORRECTION (Opus verify tay, sau khi executor báo PARTIAL)

Kết luận của executor ở trên **sai về cơ chế**, dù đúng về con số. Đã chạy lại probe mở rộng để phân biệt hai giả thuyết: (a) tag bị `optimize()` xoá, hay (b) `optimize()` từ chối chạy khi thấy tag.

```
rows_latest 2
optimize RAISED: RuntimeError lance error: Cleanup error: 1 tagged version(s) have been marked for cleanup...
A) checkout(tag) after failed optimize rows = 1     ← data cũ VẪN ĐỌC ĐƯỢC
rows_latest_again 2
optimize() no-arg RAN ok                            ← compaction-only chạy bình thường
B) checkout(tag) after compaction-only rows = 1     ← data cũ VẪN ĐỌC ĐƯỢC
C) after tag delete, cleanup RAN ok -> tag WAS the blocker
tags_now {}
```

Giả thuyết (b) đúng. `optimize(cleanup_older_than=...)` **fail-safe**: raise TRƯỚC khi chạm data, không xoá gì. Tag + version cũ nguyên vẹn sau cả hai đường (`cleanup` raise, và `optimize()` compaction-only). Xoá tag đi thì cleanup chạy được → tag đúng là thứ chặn.

Sửa hai claim sai trong báo cáo executor:
1. *"Tags survive by accident, not by design"* — sai. Fail-safe (raise trước khi ghi) là thiết kế có chủ đích, không phải tai nạn.
2. *"The inner async method accepts `error_if_tagged_old_versions`"* — sai. Signature thật của cả hai tầng:
   - `LanceTable.optimize(self, *, cleanup_older_than, delete_unverified, retrain)`
   - `AsyncTable.optimize(self, *, cleanup_older_than, delete_unverified, retrain)`

   Không tầng nào có `error_if_tagged_old_versions`. Đó là option ở tầng `lance` dataset, `lancedb` 0.29.2 Python không expose. Thông báo lỗi gợi ý một cờ không gọi tới được từ API này.

### Gate sau correction

| Gate | Trạng thái | Căn cứ |
|---|---|---|
| G1 — release cũ đọc được sau `optimize` | **PASS** | `checkout("rel-A")` = 1 row ở cả hai nhánh (A và B) |
| G1b — `restore(tag_string)` | **PASS** | rows_after_restore = 1 |

§0.2 giữ nguyên: *"A version with a tag pointing at it is retained regardless of age"* — đúng về kết quả. Cơ chế khác plan hình dung (refuse-to-run thay vì skip-and-continue), nhưng hệ quả an toàn hơn: không thể xoá nhầm release đã tag.

### Bug thiết kế API phát hiện được (phải sửa TRƯỚC khi code M1)

Plan §M1 dòng 334-338 định nghĩa:
```python
def maintain(table, keep_days: int = 30) -> dict:
    """table.optimize(cleanup_older_than=timedelta(days=keep_days))..."""
```
Hàm này sẽ **luôn raise** trên bất kỳ table có ≥1 tag — tức mọi table production, vì kiến trúc §0.2 tag từng release. Đây không phải rollback §5, mà là lỗi trong đặc tả API cần sửa trước khi viết `lancedb_versioning.py`.

## Plan Implications

- **Section 0.2 decision ("LanceDB native version+tag instead of directory-per-release")**: Still viable. `restore(tag)` works. Tags are the mechanism.
- **Section 3.3 item 1 (TAG_SURVIVES_OPTIMIZE)**: FAIL. The plan's assumption that `optimize(cleanup_older_than=timedelta(0))` can be called safely while preserving tags is wrong for lancedb 0.29.2. The sync wrapper lacks `error_if_tagged_old_versions`.
- **G1/G1b gates**: G1 depends on this probe. Since TAG_SURVIVES_OPTIMIZE cannot pass with the current API, G1 should FAIL unless the mitigation in **M1-R1** is adopted (never pass `cleanup_older_than`; see `data_lifecycle_plan_v2.md:375`, M1-R1 risk row — NOT Phụ lục B, which is only the Day-1 probe checklist).
- **M1-R1 mitigation**: "never pass cleanup_older_than" — this is the only path forward with lancedb 0.29.2 sync API.
