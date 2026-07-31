"""TDD: rag2025/src/infrastructure/lancedb_versioning.py — LanceDB native
version/tag wrapper (M1). Talks to a REAL temporary lancedb 0.29.2 table
under D:/tmp — no mocks — because M1-T3 is the architecture gate probe
that decided §0.2 (native version+tag over directory-per-release).

Ground truth (docs/evidence/M1-tag-durability.md, Opus CORRECTION section):
optimize(cleanup_older_than=...) RAISES on any tagged table — fail-safe,
raises before touching data, nothing lost. maintain() below therefore never passes
cleanup_older_than, so it must succeed (not raise) even on a tagged table.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import lancedb
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.lancedb_versioning import (  # noqa: E402
    maintain,
    rollback_to,
    tag_release,
)


@pytest.fixture
def temp_table():
    Path("D:/tmp").mkdir(parents=True, exist_ok=True)
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="m1_versioning_probe_")
    db = lancedb.connect(d)
    t = db.create_table("probe", data=[{"id": "a", "v": [0.1, 0.2, 0.3, 0.4]}])
    yield t


def test_tag_survives_optimize(temp_table):
    """M1-T3 — the gate. See CORRECTION in docs/evidence/M1-tag-durability.md."""
    t = temp_table
    tag_release(t, "A")
    t.add([{"id": "b", "v": [0.5, 0.6, 0.7, 0.8]}])

    maintain(t)  # must NOT raise

    t.checkout("release-A")
    assert t.count_rows() == 1
    t.checkout_latest()
    assert t.count_rows() == 2


def test_restore_by_tag_name(temp_table):
    """M1-T4."""
    t = temp_table
    tag_release(t, "A")
    t.add([{"id": "b", "v": [0.5, 0.6, 0.7, 0.8]}])
    assert t.count_rows() == 2

    new_version = rollback_to(t, "A")
    assert t.count_rows() == 1
    assert isinstance(new_version, int)


def test_promote_prod_idempotent(temp_table):
    """M1-T5."""
    from src.infrastructure.lancedb_versioning import promote_to_prod

    t = temp_table
    tag_release(t, "A")
    v1 = promote_to_prod(t, "A")
    v2 = promote_to_prod(t, "A")
    assert v1 == v2
    tags = t.tags.list()
    assert tags["prod"]["version"] == v1


def test_prune_versions_deletes_tag_before_cleanup(temp_table):
    """Extra coverage (non-Tn) for prune_versions — required by the API
    spec (plan lines 343-347) and the no-stub standing constraint."""
    from src.infrastructure.lancedb_versioning import prune_versions

    t = temp_table
    tag_release(t, "OLD")
    t.add([{"id": "b", "v": [0.5, 0.6, 0.7, 0.8]}])

    result = prune_versions(t, before_version=t.version)

    assert "release-OLD" not in t.tags.list()
    assert "release-OLD" in result["deleted_tags"]


def test_read_only_at_restores_latest_on_exit(temp_table):
    """Extra coverage (non-Tn) for read_only_at — required by the API
    spec (plan lines 331-333) and the no-stub standing constraint."""
    from src.infrastructure.lancedb_versioning import read_only_at

    t = temp_table
    tag_release(t, "A")
    t.add([{"id": "b", "v": [0.5, 0.6, 0.7, 0.8]}])

    with read_only_at(t, "A") as audit_view:
        assert audit_view.count_rows() == 1
    assert t.count_rows() == 2
