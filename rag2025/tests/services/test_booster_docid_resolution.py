# @spec(TICKET-45) test_booster_docid_resolution — symbolic-id resolution
"""Ticket 45 (Loi B): the aggregation booster references SYMBOLIC chunk ids
(``tuyensinh_overview_2026_v2``) that live in the LanceDB ``doc_id`` column,
while ``fetch_by_id`` historically queried only the hash ``id`` column — so
every booster ref missed and the booster silently de-boosted
(``BOOSTER_CHUNK_MISS``).

Pinned here:
  1. fetch_by_id resolves a symbolic id on the LIVE table (pinned year 2026)
     and returns it with chunk_id == the id that was asked (symbolic in,
     symbolic out).
  2. boost_with_aggregation actually injects (> 0 docs) for
     "HUSC đào tạo những ngành nào" at year=2026, and BOOSTER_CHUNK_MISS does
     NOT increase for the 3 refs the query resolves.
  3. When one logical doc_id matches several physical rows (measured: up to 3
     on the live table), the promoted row is a deterministic function of the
     row set — reordering the input returns the same winner — and the rule
     distinguishes the real content difference (28-nganh vs 29-nganh/CTĐT).
  4. seen_ids and inject_ids share one space: when the promoted physical
     chunk is already in the baseline (as a hash-keyed vector-search doc),
     the booster RE-TAGS that copy (booster_injected=True) instead of
     injecting a duplicate. Mutation target: remove the retag branch and
     this test goes RED.
  5. The hash path of fetch_by_id (GraphRAG single-fetch contract) is
     unchanged: a physical id resolves to the row's own chunk_id, and the
     batch fetch_by_ids never resolves symbolic ids.

Year pinning: CURRENT_ADMISSION_YEAR=2026 is set DELIBERATELY (matches prod
data ingestion), mirroring tests/services/test_booster_chunkid_valid.py —
without it the module-level _YEAR follows the system clock and the test would
track the calendar instead of the data.
"""
from __future__ import annotations

import importlib
import itertools
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

RAG_ROOT = Path(__file__).resolve().parents[2]
SRC = RAG_ROOT / "src"
for p in (str(RAG_ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Pin the admission year BEFORE importing the booster/rules modules so the
# module-level _YEAR snapshot is 2026 (deliberate pin, see module docstring).
os.environ["CURRENT_ADMISSION_YEAR"] = "2026"
os.environ.pop("APP_ENV", None)

from src.services import aggregation_rules, temporal_authority  # noqa: E402
importlib.reload(temporal_authority)
importlib.reload(aggregation_rules)
from src.services import aggregation_booster  # noqa: E402
importlib.reload(aggregation_booster)

from src.services.aggregation_booster import (  # noqa: E402
    boost_with_aggregation,
    detect_aggregation_chunks,
)
from src.services.lancedb_retrieval import (  # noqa: E402
    LanceDBRetriever,
    RetrievedDocument,
    _pick_promoted_row,
)


def _miss_count() -> int:
    """Current BOOSTER_CHUNK_MISS value, read through the MODULE object.

    tests/services/conftest.py reloads src.services.aggregation_booster
    before every test, which rebinds the module global to a FRESH counter
    instance. A name imported once at collection time would then observe a
    stale object (always 0) and the assertion would pass vacuously. The
    module object itself survives reloads, so attribute lookup at call time
    always sees the live counter.
    """
    return aggregation_booster.BOOSTER_CHUNK_MISS.get()


# The three booster refs of the pinned query that are ambiguous on the live
# table (one logical doc_id -> several physical rows). Measured live:
#   liet_ke_28_nganh_2026_v2            -> 3 rows
#   bang_xep_hang_chi_tieu_2026_v2      -> 2 rows
#   tuyensinh_overview_2026_v2          -> 2 rows
AMBIGUOUS_REFS = [
    "liet_ke_28_nganh_2026_v2",
    "bang_xep_hang_chi_tieu_2026_v2",
    "tuyensinh_overview_2026_v2",
]
NEUTRAL_QUERY = "HUSC đào tạo những ngành nào"


# ── Live-table fixture (skip if unavailable, assert hard when present) ─────
@pytest.fixture(scope="module")
def live_retriever_and_df():
    env_path = RAG_ROOT / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)
    try:
        import lancedb
        from config.settings import RAGSettings
        settings = RAGSettings()
        db = lancedb.connect(settings.LANCEDB_URI)
        table_list = db.list_tables()
        actual_tables = getattr(table_list, "tables", []) or []
        if not actual_tables:
            try:
                for item in table_list:
                    if isinstance(item, tuple) and item[0] == "tables":
                        actual_tables = item[1]
                        break
            except Exception:
                pass
        if settings.LANCEDB_TABLE not in actual_tables:
            pytest.skip(f"LanceDB table {settings.LANCEDB_TABLE!r} not present")
        retriever = LanceDBRetriever.from_env()
        table = db.open_table(settings.LANCEDB_TABLE)
        df = table.to_pandas()
        df["id"] = df["id"].astype(str)
        if "doc_id" in df.columns:
            df["doc_id"] = df["doc_id"].astype("string")
        return retriever, df
    except Exception as e:  # pragma: no cover - environment guard
        pytest.skip(f"LanceDB unavailable: {e}")


# ── Fake adapter/retriever for the pure-logic tests ────────────────────────
class _FakeAdapter:
    """Serves canned rows for both id spaces, recording the calls."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows
        self.id_calls: List[List[str]] = []
        self.doc_id_calls: List[List[str]] = []

    def ensure_table(self) -> None:
        return None

    def fetch_rows_by_ids(self, ids):
        self.id_calls.append(list(ids))
        wanted = set(ids)
        return [r for r in self.rows if str(r.get("id")) in wanted]

    def fetch_rows_by_doc_ids(self, ids):
        self.doc_id_calls.append(list(ids))
        wanted = set(ids)
        return [r for r in self.rows if r.get("doc_id") in wanted]


class _FakeRetriever:
    """Minimal stand-in exposing the same fetch surface the booster uses."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._adapter = _FakeAdapter(rows)

    def fetch_by_id(self, chunk_id: str) -> Optional[RetrievedDocument]:
        rows = self._adapter.fetch_rows_by_ids([chunk_id])
        via = "id"
        if not rows:
            rows = self._adapter.fetch_rows_by_doc_ids([chunk_id])
            via = "doc_id"
        if not rows:
            return None
        row = _pick_promoted_row(rows) if len(rows) > 1 else rows[0]
        return RetrievedDocument(
            text=row.get("text", ""),
            source=row.get("source"),
            chunk_id=(
                chunk_id if via == "doc_id"
                else (row.get("chunk_id") or row.get("id") or "")
            ),
            metadata={},
            score=1.0,
            point_id=str(row.get("id")),
        )

    def fetch_by_ids(self, chunk_ids):
        rows = self._adapter.fetch_rows_by_ids(chunk_ids)
        return {
            str(r["id"]): RetrievedDocument(
                text=r.get("text", ""),
                source=r.get("source"),
                chunk_id=r.get("chunk_id") or str(r["id"]),
                metadata={},
                score=1.0,
                point_id=str(r["id"]),
            )
            for r in rows
        }


def _row(id_: str, doc_id: Optional[str] = None, text: str = "", **extra) -> Dict[str, Any]:
    row = {
        "id": id_,
        "chunk_id": id_,
        "doc_id": doc_id,
        "text": text or f"text-{id_}",
        "source": "synthetic",
        "metadata_json": "{}",
    }
    row.update(extra)
    return row


# ── Criterion 1: symbolic id resolves on the LIVE table ───────────────────
def test_fetch_by_id_resolves_symbolic_ref_on_live_table(live_retriever_and_df):
    retriever, _ = live_retriever_and_df
    doc = retriever.fetch_by_id("tuyensinh_overview_2026_v2")
    assert doc is not None, (
        "BUG: fetch_by_id cannot resolve the symbolic booster ref — "
        "the booster silently de-boosts (BOOSTER_CHUNK_MISS)"
    )
    # Symbolic in == symbolic out: the returned chunk_id is the id asked for.
    assert doc.chunk_id == "tuyensinh_overview_2026_v2"
    assert doc.text, "resolved doc must carry the chunk text"
    assert doc.point_id, "resolved doc must expose the physical row id"


def test_fetch_by_id_hash_path_semantics_unchanged(live_retriever_and_df):
    """GraphRAG single-fetch contract: a physical (hash) id resolves to the
    row's OWN chunk_id — not to some symbolic id."""
    retriever, df = live_retriever_and_df
    hash_id = str(df["id"].iloc[0])
    doc = retriever.fetch_by_id(hash_id)
    assert doc is not None
    assert doc.chunk_id == hash_id
    assert doc.point_id == hash_id


def test_fetch_by_ids_batch_path_never_resolves_symbolic(live_retriever_and_df):
    """Criterion 5 companion: the batch/hash path keeps its id-space contract
    (GraphRAG expander passes hash ids; symbolic refs are absent, not
    silently reinterpreted)."""
    retriever, _ = live_retriever_and_df
    out = retriever.fetch_by_ids(["tuyensinh_overview_2026_v2"])
    assert out == {}


# ── Criterion 2: booster injects, no new misses (live) ────────────────────
def test_year_is_pinned_to_2026_for_booster_modules():
    assert int(temporal_authority.get_current_admission_year()) == 2026
    assert aggregation_booster._YEAR == 2026


def test_boost_injects_over_live_table_without_new_misses(live_retriever_and_df):
    retriever, _ = live_retriever_and_df
    refs = detect_aggregation_chunks(NEUTRAL_QUERY)
    assert refs, "detector must fire for the pinned enumeration query"

    # Every ref the detector produced must resolve (no silent misses).
    miss_before = _miss_count()
    resolved = {cid: retriever.fetch_by_id(cid) for cid in refs}
    unresolved = [cid for cid, d in resolved.items() if d is None]
    assert _miss_count() == miss_before, (
        "fetch_by_id must not bump the miss counter for refs that resolve"
    )
    # The 3 ambiguous refs from the ticket MUST be among the resolved refs.
    for cid in AMBIGUOUS_REFS:
        assert cid in refs, f"detector lost ref {cid} (refs={refs})"
        assert resolved[cid] is not None, f"{cid} did not resolve"
        assert resolved[cid].chunk_id == cid

    # Baseline: 3 physical rows that are NOT any promoted winner, so the
    # injection path (not the retag path) is what we measure.
    winner_points = {d.point_id for d in resolved.values() if d is not None}
    pool = [r for r in retriever._adapter.fetch_rows_by_ids(
        [str(i) for i in _live_nonwinner_ids(resolved)]
    )]
    baseline = retriever._normalize_rows(pool)
    assert len(baseline) == 3
    assert all(d.point_id not in winner_points for d in baseline)

    miss_before = _miss_count()
    boosted = boost_with_aggregation(
        query=NEUTRAL_QUERY,
        baseline_docs=baseline,
        lancedb_retriever=retriever,
        top_k=15,
        max_inject=2,
    )
    miss_after = _miss_count()

    injected = [d for d in boosted if (d.metadata or {}).get("booster_injected")]
    assert len(injected) > 0, (
        "booster injected nothing — the silent-de-boost bug is alive"
    )
    assert miss_after == miss_before, (
        f"BOOSTER_CHUNK_MISS grew by {miss_after - miss_before} "
        f"for refs {unresolved}"
    )
    # Injected docs carry SYMBOLIC ids (same space as inject_ids).
    assert all(d.chunk_id in refs for d in injected)


def _live_nonwinner_ids(resolved: Dict[str, RetrievedDocument]) -> List[str]:
    """First 3 physical row ids that are not promoted winners for the refs."""
    import lancedb
    from config.settings import RAGSettings
    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.open_table(settings.LANCEDB_TABLE)
    banned = {d.point_id for d in resolved.values()}
    out: List[str] = []
    for row in table.search().limit(50).to_list():
        rid = str(row.get("id"))
        if rid not in banned:
            out.append(rid)
        if len(out) == 3:
            break
    return out


# ── Criterion 3: deterministic promoted-row selection ─────────────────────
def test_pick_promoted_row_is_input_order_independent(live_retriever_and_df):
    """Reorder the input rows -> same winner. The three real rows for
    liet_ke_28_nganh_2026_v2 differ in CONTENT (28-nganh vs 29-nganh/CTĐT),
    so an order-dependent pick would be observable."""
    retriever, _ = live_retriever_and_df
    rows = retriever._adapter.fetch_rows_by_doc_ids(["liet_ke_28_nganh_2026_v2"])
    assert len(rows) >= 3, (
        "fixture assumption broken: expected >=3 physical rows for the "
        "ambiguous ref on the live table"
    )
    winners = set()
    for perm in itertools.permutations(rows):
        winners.add(str(_pick_promoted_row(list(perm))["id"]))
    assert len(winners) == 1, f"winner depends on input order: {winners}"
    winner = _pick_promoted_row(rows)
    # The winner is the 29-ngành/CTĐT content row, NOT the legacy 28-ngành
    # row — and the test distinguishes the two (real content difference).
    texts = {str(r["id"]): str(r.get("text", "")) for r in rows}
    assert "29 ngành/CTĐT" in texts[str(winner["id"])], (
        f"winner text mismatch: {texts[str(winner['id'])][:80]!r}"
    )
    assert any("28 ngành" in t for t in texts.values()), (
        "fixture rows no longer contain the 28-ngành variant — the "
        "distinction assertion above would be vacuous"
    )


def test_pick_promoted_rule_tiers():
    """The documented rule: non-legacy > largest data_year > smallest id."""
    # Tier 1: a non-legacy row wins even with an id that sorts last.
    rows = [
        _row("0000", doc_id="d", release_id="legacy", data_year=2099),
        _row("zzzz", doc_id="d", release_id="canonical_fix_v9", data_year=2020),
    ]
    assert _pick_promoted_row(rows)["id"] == "zzzz"

    # Tier 2: among non-legacy, largest data_year wins.
    rows = [
        _row("aaaa", doc_id="d", release_id="fix", data_year=2026),
        _row("bbbb", doc_id="d", release_id="fix", data_year=2027),
    ]
    assert _pick_promoted_row(rows)["id"] == "bbbb"

    # Tier 3: full tie -> smallest id lexicographically (stable, no scan order).
    rows = [
        _row("ccc", doc_id="d", release_id="fix", data_year=2026),
        _row("aaa", doc_id="d", release_id="fix", data_year=2026),
        _row("bbb", doc_id="d", release_id="fix", data_year=2026),
    ]
    for perm in itertools.permutations(rows):
        assert _pick_promoted_row(list(perm))["id"] == "aaa"


# ── Criterion 4: point-space retag (mutation target) ──────────────────────
def test_point_space_retag_fires_when_baseline_holds_the_physical_chunk():
    """Baseline contains the promoted row as a hash-keyed vector-search doc.
    The booster must RE-TAG that copy (booster_injected=True, chunk_id
    normalized to the symbolic ref) instead of injecting a duplicate — and
    must NOT consume a max_inject slot for it."""
    winner_hash = "33544fb8f9fb1711"
    sym = "liet_ke_28_nganh_2026_v2"
    rows = [
        _row(winner_hash, doc_id=sym, text="Danh sách đầy đủ 29 ngành/CTĐT"),
        _row("other1", doc_id="tuyensinh_overview_2026_v2", text="overview row"),
        _row("other2", doc_id="bang_xep_hang_chi_tieu_2026_v2", text="ranking row"),
    ]
    fake = _FakeRetriever(rows)
    baseline = [
        RetrievedDocument(text="Danh sách đầy đủ 29 ngành/CTĐT", source="synthetic",
                          chunk_id=winner_hash, metadata={}, score=0.42,
                          point_id=winner_hash),
        RetrievedDocument(text="unrelated", source="synthetic",
                          chunk_id="unrelated", metadata={}, score=0.40,
                          point_id="unrelated"),
    ]
    # Force the detector to the single symbolic ref that matches the baseline.
    out = _boost_with_single_ref(fake, baseline, sym)
    tagged = [d for d in out if (d.metadata or {}).get("booster_injected")]
    assert len(tagged) == 3, f"retag + 2 injects expected, got {len(tagged)}"
    doc = next(d for d in tagged if d.point_id == winner_hash)
    # The RETAGGED BASELINE copy: score preserved (0.42), not the injected
    # copy's sentinel (1.0).
    assert doc.score == 0.42, (
        "baseline copy was not retagged — an injected duplicate replaced it "
        "(seen_ids/inject_ids spaces diverged)"
    )
    assert doc.chunk_id == sym, "retagged doc must normalize to the symbolic id"
    # No duplicate: exactly one doc carries the physical point.
    assert sum(1 for d in out if d.point_id == winner_hash) == 1
    # Retag does not consume the max_inject slot: the other two refs still
    # inject (capped at 2).
    injected_ids = {d.chunk_id for d in out
                    if (d.metadata or {}).get("booster_injected") and d.score == 1.0}
    assert len(injected_ids) == 2, (
        f"retag consumed an inject slot: injected={injected_ids}"
    )


def _boost_with_single_ref(fake, baseline, sym: str):
    """Run boost_with_aggregation with the detector pinned to [sym] + two
    other refs, so the test exercises exactly one point-space retag."""
    import src.services.aggregation_booster as B
    original = B.detect_aggregation_chunks
    B.detect_aggregation_chunks = lambda q: [
        sym, "tuyensinh_overview_2026_v2", "bang_xep_hang_chi_tieu_2026_v2",
    ]
    try:
        return B.boost_with_aggregation(
            query="HUSC đào tạo những ngành nào",
            baseline_docs=baseline,
            lancedb_retriever=fake,
            top_k=15,
            max_inject=2,
        )
    finally:
        B.detect_aggregation_chunks = original


def test_symbolic_space_retag_baseline_still_works():
    """Companion of tests/services/test_booster_retag_baseline.py: when the
    baseline already carries the SYMBOLIC id, the doc is tagged in place and
    no fetch happens."""
    sym = "tuyensinh_overview_2026_v2"
    fake = _FakeRetriever([_row("h1", doc_id=sym, text="x")])
    baseline = [RetrievedDocument(text="x", source="s", chunk_id=sym,
                                  metadata={}, score=0.9, point_id="h1")]
    import src.services.aggregation_booster as B
    original = B.detect_aggregation_chunks
    B.detect_aggregation_chunks = lambda q: [sym]
    try:
        out = B.boost_with_aggregation(
            query="q", baseline_docs=baseline, lancedb_retriever=fake,
            top_k=5, max_inject=2,
        )
    finally:
        B.detect_aggregation_chunks = original
    assert len(out) == 1
    assert out[0].metadata.get("booster_injected") is True
    assert fake._adapter.doc_id_calls == [], "symbolic hit must not fetch"


# ── Cap/retag ordering: retag must survive a full inject cap ─────────────
def test_retag_fires_for_baseline_resident_ref_after_cap_is_full():
    """Production bug this pins: the old loop `break`ed once the cap was
    full, so a LATER ref whose chunk already sits in the baseline (point
    space) was never fetched and never got its fusion-protection retag.
    With the fetch-then-gate restructure the scan continues and the
    non-consuming retag still fires.

    Detector order [A, B, C]: A+B fill the cap (not in baseline), C's
    physical chunk IS in the baseline. max_inject=2.
    Old code: break before C → C untagged → RED. New: C retagged → GREEN."""
    rows = [
        _row("hA", doc_id="ref_a_2026", text="A"),
        _row("hB", doc_id="ref_b_2026", text="B"),
        _row("hC", doc_id="ref_c_2026", text="C"),
    ]
    fake = _FakeRetriever(rows)
    baseline = [
        RetrievedDocument(text="C", source="synthetic", chunk_id="hC",
                          metadata={}, score=0.5, point_id="hC"),
    ]
    import src.services.aggregation_booster as B
    original = B.detect_aggregation_chunks
    B.detect_aggregation_chunks = lambda q: [
        "ref_a_2026", "ref_b_2026", "ref_c_2026",
    ]
    try:
        out = B.boost_with_aggregation(
            query="q", baseline_docs=baseline, lancedb_retriever=fake,
            top_k=15, max_inject=2,
        )
    finally:
        B.detect_aggregation_chunks = original
    tagged = [d for d in out if (d.metadata or {}).get("booster_injected")]
    # A + B injected (score sentinel 1.0), C retagged in place (score 0.5).
    assert len(tagged) == 3, (
        f"cap-full baseline ref lost its retag: tagged={[d.chunk_id for d in tagged]}"
    )
    cdoc = next(d for d in tagged if d.point_id == "hC")
    assert cdoc.score == 0.5, "retagged baseline copy must keep its rerank score"
    assert cdoc.chunk_id == "ref_c_2026", "retag must normalize to the symbolic id"
    assert sum(1 for d in out if d.point_id == "hC") == 1, "no duplicate for hC"
    injected = [d for d in tagged if d.score == 1.0]
    assert [d.chunk_id for d in injected] == ["ref_a_2026", "ref_b_2026"]


def test_booster_chunk_miss_is_gated_by_remaining_cap():
    """BOOSTER_CHUNK_MISS counts refs that COULD have been injected but
    weren't. Dead ref BEFORE the cap fills → counted (matches old code).
    Dead ref AFTER the cap fills → not counted: the booster already
    delivered max_inject docs, so nothing was silently de-boosted (and the
    old fetch-then-break loop never fetched it at all — same observable
    count)."""
    rows = [
        _row("hA", doc_id="ref_a_2026", text="A"),
        _row("hB", doc_id="ref_b_2026", text="B"),
    ]
    import src.services.aggregation_booster as B
    original = B.detect_aggregation_chunks
    try:
        # Control: dead ref first (cap not full) → exactly one miss.
        B.detect_aggregation_chunks = lambda q: [
            "ref_dead", "ref_a_2026", "ref_b_2026",
        ]
        fake = _FakeRetriever(rows)
        before = _miss_count()
        out = B.boost_with_aggregation(
            query="q", baseline_docs=[], lancedb_retriever=fake,
            top_k=15, max_inject=2,
        )
        assert _miss_count() == before + 1, (
            "dead ref while slots remain must count as a miss"
        )
        assert sum(1 for d in out if (d.metadata or {}).get("booster_injected")) == 2

        # Gated case: dead ref LAST (cap already full) → no extra miss.
        B.detect_aggregation_chunks = lambda q: [
            "ref_a_2026", "ref_b_2026", "ref_dead",
        ]
        fake = _FakeRetriever(rows)
        before = _miss_count()
        out = B.boost_with_aggregation(
            query="q", baseline_docs=[], lancedb_retriever=fake,
            top_k=15, max_inject=2,
        )
        assert _miss_count() == before, (
            "dead ref after the cap is full must not inflate the miss counter"
        )
        assert sum(1 for d in out if (d.metadata or {}).get("booster_injected")) == 2
    finally:
        B.detect_aggregation_chunks = original
