"""TDD: Server-side `fetch_rows_by_ids` filter on `LanceDBAdapter`.

Contract pinned by these tests:
  1. `fetch_rows_by_ids` builds a SAFE `id IN (...)` predicate with
     SQL-style single-quote escaping (`'` → `''`).
  2. Empty/None input → `[]` without touching the table.
  3. Non-string ids are silently skipped.
  4. The implementation NEVER falls back to a full `to_pandas()` scan
     (a fake table whose `to_pandas` raises is used to enforce this).
  5. The returned row shape matches what `lancedb_retrieval.fetch_by_ids`
     expects to build `RetrievedDocument`s identical to the prior
     in-memory filter path (text/source/chunk_id/metadata_json/id).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infrastructure.lancedb_adapter import LanceDBAdapter  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Fake table: chains `search().where(predicate).limit(n).to_list()`
# ─────────────────────────────────────────────────────────────────────────────


class _FakeSearch:
    """Mimics the lancedb search-builder: `search().where(pred).limit(n).to_list()`.

    The `where` predicate is parsed for an `id IN (...)` clause; the
    matching id set is then intersected with the canned rows. Any
    rows whose `id` is in that set come back (up to `limit`)."""

    _IN_RE = None  # compiled lazily

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = rows
        self.predicate: Optional[str] = None
        self.limit_value: Optional[int] = None
        self.to_list_calls = 0

    def where(self, predicate: str) -> "_FakeSearch":
        self.predicate = predicate
        return self

    def limit(self, n: int) -> "_FakeSearch":
        self.limit_value = n
        return self

    def to_list(self) -> List[Dict[str, Any]]:
        self.to_list_calls += 1
        matching = set(self._parse_in_ids(self.predicate or ""))
        out = [r for r in self._rows if r.get("id") in matching]
        if self.limit_value is not None:
            out = out[: self.limit_value]
        return out

    @staticmethod
    def _parse_in_ids(predicate: str) -> List[str]:
        """Best-effort parse of `id IN ('a', 'b''c', ...)` for the fake."""
        import re

        m = re.search(r"id\s+IN\s*\((.*)\)\s*$", predicate.strip(), re.IGNORECASE | re.DOTALL)
        if not m:
            return []
        body = m.group(1)
        ids: List[str] = []
        i, n = 0, len(body)
        while i < n:
            ch = body[i]
            if ch.isspace() or ch == ",":
                i += 1
                continue
            if ch != "'":
                i += 1
                continue
            # Parse a SQL single-quoted string with '' as escape.
            i += 1
            buf: List[str] = []
            while i < n:
                if body[i] == "'":
                    if i + 1 < n and body[i + 1] == "'":
                        buf.append("'")
                        i += 2
                        continue
                    i += 1  # closing quote
                    break
                buf.append(body[i])
                i += 1
            ids.append("".join(buf))
        return ids


class _FakeTable:
    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._search = _FakeSearch(rows)
        self._rows = list(rows)
        # Forbid any full-table scan in this path.
        self.to_pandas = MagicMock(
            side_effect=AssertionError(
                "LanceDBAdapter.fetch_rows_by_ids MUST NOT call to_pandas(); "
                "a full-table scan defeats the latency fix"
            )
        )

    def search(self, *args, **kwargs) -> _FakeSearch:
        # No-vector search() is the pattern used by fetch_rows_by_ids
        # (lancedb 0.29.x allows it; only the .where() + .limit() + .to_list()
        # chain matters here).
        return self._search

    @property
    def last_predicate(self) -> Optional[str]:
        return self._search.predicate

    @property
    def last_limit(self) -> Optional[int]:
        return self._search.limit_value

    @property
    def search_calls(self) -> int:
        return 1 if self._search.predicate is not None else 0


def _make_adapter(rows: List[Dict[str, Any]]) -> tuple[LanceDBAdapter, _FakeTable]:
    """Build an adapter that uses the fake table directly (skip __init__)."""
    fake_table = _FakeTable(rows)
    adapter = LanceDBAdapter.__new__(LanceDBAdapter)
    adapter._uri = "memory://"
    adapter._table_name = "t"
    adapter._db = MagicMock()
    adapter._table = fake_table
    return adapter, fake_table


def _row(cid: str, text: str = "", source: str = "synthetic",
         meta: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "id": cid,
        "chunk_id": cid,
        "text": text or f"text-{cid}",
        "source": source,
        "metadata_json": json.dumps(meta or {"data_year": 2026}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Part A — Adapter unit (predicate + no-scan)
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_rows_by_ids_empty_list_returns_empty_without_querying():
    adapter, fake_table = _make_adapter([_row("c1"), _row("c2")])
    assert adapter.fetch_rows_by_ids([]) == []
    assert fake_table.search_calls == 0, "empty list must not hit the table"


def test_fetch_rows_by_ids_none_input_returns_empty_without_querying():
    adapter, fake_table = _make_adapter([_row("c1")])
    assert adapter.fetch_rows_by_ids(None) == []  # type: ignore[arg-type]
    assert fake_table.search_calls == 0


def test_fetch_rows_by_ids_builds_safe_in_predicate_with_quoted_ids():
    adapter, fake_table = _make_adapter([_row("c1"), _row("c2"), _row("c3")])

    adapter.fetch_rows_by_ids(["c1", "c2", "c3"])

    pred = fake_table.last_predicate
    assert pred is not None
    # Each id must appear wrapped in single quotes, comma-separated.
    assert pred == "id IN ('c1', 'c2', 'c3')", f"unexpected predicate: {pred!r}"


def test_fetch_rows_by_ids_escapes_single_quote_via_sql_doubling():
    """An id containing a single quote MUST be escaped (`'` → `''`) so it
    cannot break out of the SQL string and inject a malicious predicate."""
    adapter, fake_table = _make_adapter([_row("evil'id")])
    adapter.fetch_rows_by_ids(["evil'id"])

    pred = fake_table.last_predicate
    # Doubled single-quote inside the literal: 'evil''id'
    assert pred == "id IN ('evil''id')", f"unsafe predicate: {pred!r}"
    # The string MUST end with the closing single quote, not a dangling
    # unescaped one (i.e. no `id IN ('evil'` injection breakout).
    assert pred.endswith("')"), f"predicate did not terminate cleanly: {pred!r}"


def test_fetch_rows_by_ids_skips_non_string_ids_silently():
    adapter, fake_table = _make_adapter([_row("c1")])
    # Mix of valid + invalid id types — invalid are dropped.
    out = adapter.fetch_rows_by_ids(["c1", None, 123, 4.5, "", "c1"])  # type: ignore[list-item]

    # Only 'c1' is a non-empty str; the rest are skipped / deduped.
    # The predicate should reference only the valid strings.
    assert "c1" in fake_table.last_predicate
    # No None/123 leakage into the predicate (would error SQL).
    assert "None" not in fake_table.last_predicate
    assert "123" not in fake_table.last_predicate


def test_fetch_rows_by_ids_empty_after_filtering_returns_empty_list():
    """If every input id is non-str, the safe-id list is empty → return []."""
    adapter, fake_table = _make_adapter([_row("c1")])
    out = adapter.fetch_rows_by_ids([None, 123, 4.5])  # type: ignore[list-item]
    assert out == []
    assert fake_table.search_calls == 0, "no valid ids → no table call"


def test_fetch_rows_by_ids_never_calls_to_pandas():
    """The whole point of this fix: NO full-table materialization."""
    adapter, fake_table = _make_adapter([_row("c1"), _row("c2")])
    adapter.fetch_rows_by_ids(["c1", "c2"])
    # The fake's to_pandas raises AssertionError on call — if the adapter
    # had fallen back to it, the test would fail with that error.
    assert fake_table.to_pandas.call_count == 0


def test_fetch_rows_by_ids_sets_limit_to_len_ids():
    adapter, fake_table = _make_adapter([_row("c1"), _row("c2"), _row("c3")])
    adapter.fetch_rows_by_ids(["c1", "c2", "c3"])
    assert fake_table.last_limit == 3


# ─────────────────────────────────────────────────────────────────────────────
# Part B — Equivalence with reference dict (RetrievedDocument field shape)
# ─────────────────────────────────────────────────────────────────────────────


def test_fetch_rows_by_ids_returns_rows_matching_the_id_set():
    """Only the rows whose id is in the requested set come back."""
    rows = [_row("c1", text="one"), _row("c2", text="two"), _row("c3", text="three")]
    adapter, _ = _make_adapter(rows)

    out = adapter.fetch_rows_by_ids(["c1", "c3", "MISSING"])

    assert {r["id"] for r in out} == {"c1", "c3"}
    by_id = {r["id"]: r for r in out}
    assert by_id["c1"]["text"] == "one"
    assert by_id["c3"]["text"] == "three"


def test_fetch_rows_by_ids_row_shape_supports_retrieveddocument_construction():
    """The dict shape returned by the adapter must contain the columns
    `fetch_by_ids` (in lancedb_retrieval) consumes to build a
    `RetrievedDocument`: id, chunk_id, text, source, metadata_json."""
    rows = [_row("c1", text="alpha", source="src-1",
                 meta={"data_year": 2026, "is_superseded": False})]
    adapter, _ = _make_adapter(rows)

    out = adapter.fetch_rows_by_ids(["c1"])

    assert len(out) == 1
    row = out[0]
    for col in ("id", "chunk_id", "text", "source", "metadata_json"):
        assert col in row, f"adapter row missing required column: {col}"
    assert row["id"] == "c1"
    assert row["chunk_id"] == "c1"
    assert row["text"] == "alpha"
    assert row["source"] == "src-1"
    # metadata_json is a string (json.loads'd downstream by fetch_by_ids).
    assert isinstance(row["metadata_json"], str)
    parsed = json.loads(row["metadata_json"])
    assert parsed == {"data_year": 2026, "is_superseded": False}


def test_fetch_rows_by_ids_equivalence_with_reference_construction():
    """The adapter output, when fed to the SAME construction logic as
    `fetch_by_ids`, produces RetrievedDocuments field-identical to a
    reference dict built directly from the canned rows.

    This is the regression guard: if the adapter column set drifts
    (e.g. drops `metadata_json`), the booster / expander would silently
    see empty metadata in production.
    """
    from src.services.lancedb_retrieval import RetrievedDocument

    rows = [
        _row("c1", text="alpha", source="src-1", meta={"k": "v1"}),
        _row("c2", text="beta", source="src-2", meta={"k": "v2"}),
    ]
    adapter, _ = _make_adapter(rows)

    out = adapter.fetch_rows_by_ids(["c1", "c2", "MISSING"])
    got = {r["id"]: r for r in out}

    # Reference path: build directly from `rows` (mimics the old
    # in-memory filter behavior that the new path must reproduce).
    ref: Dict[str, RetrievedDocument] = {}
    for r in rows:
        meta = json.loads(r["metadata_json"]) if r.get("metadata_json") else {}
        ref[r["id"]] = RetrievedDocument(
            text=r.get("text", ""),
            source=r.get("source"),
            chunk_id=r.get("chunk_id", r["id"]),
            metadata=meta,
            score=1.0,
            point_id=str(r["id"]),
        )

    # Same set of keys.
    assert set(got.keys()) == set(ref.keys()), (
        f"adapter returned {set(got.keys())} vs reference {set(ref.keys())}"
    )
    for cid in ref:
        adapter_row = got[cid]
        adapter_meta = (
            json.loads(adapter_row["metadata_json"])
            if adapter_row.get("metadata_json") else {}
        )
        # Field-by-field equality on the inputs that build a RetrievedDocument.
        assert adapter_row["text"] == ref[cid].text
        assert adapter_row["source"] == ref[cid].source
        assert adapter_row.get("chunk_id", adapter_row["id"]) == ref[cid].chunk_id
        assert adapter_meta == ref[cid].metadata
