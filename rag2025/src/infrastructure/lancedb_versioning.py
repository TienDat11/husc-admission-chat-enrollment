"""LanceDB native version/tag wrapper. See data_lifecycle_plan_v2.md M1.

CORRECTION (probe P1, docs/evidence/M1-tag-durability.md, Opus verify):
lancedb 0.29.2's sync LanceTable.optimize() does not expose
error_if_tagged_old_versions, so calling optimize(cleanup_older_than=...)
on ANY tagged table ALWAYS raises RuntimeError("Cleanup error: N tagged
version(s)...") -- fail-safe, before touching data, nothing lost. maintain()
below therefore NEVER passes cleanup_older_than; version pruning lives in
prune_versions() (Slice 8), which deletes tags first.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta


def tag_release(table, release_id: str) -> int:
    """Gắn tag 'release-{release_id}' vào version hiện tại của table.
    Tra về version number vừa được gắn tag."""
    version = table.version
    table.tags.create(f"release-{release_id}", version)
    return version


def rollback_to(table, release_id: str) -> int:
    """table.restore(f'release-{release_id}') -- tao version MOI co data cu,
    KHONG copy data (verified: docs.lancedb.com/tables/versioning). Tra ve
    version moi (table.version sau restore)."""
    table.restore(f"release-{release_id}")
    return table.version


def maintain(table) -> dict:
    """table.optimize(delete_unverified=False) -- CHI compaction, KHONG BAO
    GIO truyen cleanup_older_than (xem CORRECTION o module docstring -- moi
    table production deu co tag, truyen cleanup_older_than se luon raise)."""
    return table.optimize(delete_unverified=False)


def promote_to_prod(table, release_id: str) -> int:
    """Di chuyển tag 'prod' sang version của tag 'release-{release_id}'.
    Idempotent: gọi 2 lần liên tiếp với cùng release_id không lỗi, cùng
    version. Dùng tags.update() nếu 'prod' đã tồn tại, tags.create() nếu
    chưa."""
    tag_name = f"release-{release_id}"
    all_tags = table.tags.list()
    target_version = all_tags[tag_name]["version"]
    if "prod" in all_tags:
        table.tags.update("prod", target_version)
    else:
        table.tags.create("prod", target_version)
    return target_version


@contextmanager
def read_only_at(table, release_id: str):
    """Context manager: checkout(f'release-{release_id}') cho audit
    read-only, tự động checkout_latest() khi thoát (cả khi có exception)."""
    table.checkout(f"release-{release_id}")
    try:
        yield table
    finally:
        table.checkout_latest()


def prune_versions(table, before_version: int) -> dict:
    """Xoá version cũ TRƯỚC before_version. BẮT BUỘC tags.delete() mọi tag
    trỏ vào version < before_version TRƯỚC khi cleanup, nếu không
    optimize(cleanup_older_than=...) raise RuntimeError (xem CORRECTION ở
    đầu file). Thao tác tách riêng, KHÔNG tự gọi trong maintain(). Trả về
    dict {deleted_tags, optimize_stats}."""
    all_tags = table.tags.list()
    deleted = []
    for tag_name, info in list(all_tags.items()):
        if info["version"] < before_version:
            table.tags.delete(tag_name)
            deleted.append(tag_name)
    stats = table.optimize(
        cleanup_older_than=timedelta(seconds=0), delete_unverified=False
    )
    return {"deleted_tags": deleted, "optimize_stats": stats}
