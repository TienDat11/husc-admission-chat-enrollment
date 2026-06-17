# @spec(S14.11) test_temporal_rollover_e2e — 10-year temporal contract
"""TDD: encode the 10-year temporal contract for a Vietnamese university-
admissions RAG.

This is the regression guard that keeps the system correct for 10 years
as the admission year rolls over. The contract is: **the SAME user
query yields the correct temporal behavior across three calendar phases**
as the data landscape evolves. Only the simulated time + data state
change — no src edits, no env edits at runtime, no code change between
phases.

Phases (all driven for the SAME query string — only time + data state change):

  P1 — Jul 2026, in-season 2026
       authority year=2026, has_data(2026)=True → IN_SEASON
       rows {2026, 2025, null-year}
       contract: drop 2025, keep 2026 + null-year

  P2 — Oct 2026, gap waiting for 2027
       authority year=2027, has_data(2027)=False → PRE_SEASON_GAP
       rows {2026, 2025} (no 2027)
       retriever contract: keep 2025 + 2026 (NOT dropped — show labeled)
       generator contract: emit canonical GAP_DISCLAIMER
                           (markers: "chưa được công bố chính thức" AND "2027")
                           + keep prior-year body

  P3 — Apr 2027, in-season 2027 with fresh data
       authority year=2027, has_data(2027)=True → IN_SEASON
       rows {2027, 2026, 2025, null-year}
       contract: drop 2026 + 2025 (now both prior), keep 2027 + null-year

KEY ASSERTIONS encoded:
  * SAME query in all 3 phases (only time + data state change).
  * P1 drops 2025 but P2 KEEPS 2025 — proves the season gate
    (don't hard-drop when current-year data is absent).
  * P3 proves the rollover: once 2027 data exists, 2026 becomes
    "old" and is dropped — WITHOUT any code edit, only the
    simulated data-state + year changed.
  * null-year (stable) chunks survive in ALL in-season phases.
  * Generator emits gap-disclaimer ONLY in P2 (PRE_SEASON_GAP path).
  * Year-parametrization: P2's disclaimer says "2027" (current year),
    not "2026" (the prior year being shown).

If a phase assertion reveals a real bug in src (e.g. P3 doesn't drop
2026), STOP and REPORT it with the exact failing assertion + file:line —
DO NOT edit src to make it pass.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest


# Make `services.*` importable regardless of pytest invocation cwd.
RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The same query is reused in every phase. Only the simulated time +
# data state change between phases. This is the S14.11 contract.
SAME_QUERY = "điểm chuẩn ngành CNTT là bao nhiêu"

# Years used across the three phases.
P1_YEAR = 2026  # authority year in Jul 2026
P2_YEAR = 2027  # authority year in Oct 2026 (gap)
P3_YEAR = 2027  # authority year in Apr 2027 (in-season 2027)

# Body text used by the mocked LLM in Phase 2 — must survive the
# disclaimer prepend (proves prior-year body is kept, not suppressed).
P2_PRIOR_BODY = (
    "Năm 2026 điểm chuẩn ngành Công nghệ thông tin (CNTT) HUSC là 18.5 "
    "(theo thông báo TB60)."
)

# Disambiguating chunk_id markers so the retriever assertions are
# readable when something is wrong.
CID_CURRENT_2026 = "chunk_current_2026"
CID_PRIOR_2025 = "chunk_prior_2025"
CID_PRIOR_2026 = "chunk_prior_2026"
CID_CURRENT_2027 = "chunk_current_2027"
CID_STABLE_NULL = "chunk_stable_null_year"


# ---------------------------------------------------------------------------
# Helpers — retriever
# ---------------------------------------------------------------------------

EMBEDDING_DIM = 4


def _make_fake_search_adapter(rows: List[Dict[str, Any]]) -> MagicMock:
    """Build a MagicMock that quacks like LanceDBAdapter for retrieve().

    No real LanceDB is touched. Returns the rows as-is (post-fetch
    hard-EXCLUDE in `_apply_temporal_filter` does the year work).
    """
    fake = MagicMock()
    fake.ensure_table = MagicMock()
    fake._table = MagicMock()  # has_current_year_data path isn't hit when ctx is stubbed

    def _record_search(query_vector, top_k, where_clause=None):
        return list(rows)

    fake.search.side_effect = _record_search
    return fake


def _make_retriever(
    rows: List[Dict[str, Any]],
    *,
    year: int,
    has_data: bool,
) -> Any:
    """Build a LanceDBRetriever bypassing __init__ and stub the season
    seam so we can simulate any (year, data-state) combination without a
    real DB or wall clock.
    """
    from services.lancedb_retrieval import LanceDBRetriever, LanceDBRetrieverConfig
    from services.season import SeasonPhase

    retriever = LanceDBRetriever.__new__(LanceDBRetriever)
    retriever._config = LanceDBRetrieverConfig(
        uri="memory://",
        table_name="t",
        embedding_dim=EMBEDDING_DIM,
        default_top_k=5,
    )
    retriever._adapter = _make_fake_search_adapter(rows)

    season = SeasonPhase.IN_SEASON if has_data else SeasonPhase.PRE_SEASON_GAP
    retriever.get_admission_context = MagicMock(  # type: ignore[assignment]
        return_value={"year": year, "has_data": has_data, "season": season}
    )
    return retriever


def _make_vector(dim: int = EMBEDDING_DIM) -> List[float]:
    return [0.0] * dim


def _row(chunk_id: str, *, data_year: Optional[int]) -> Dict[str, Any]:
    """Build a synthetic row shaped like one returned by LanceDBAdapter.search.

    `data_year=None` simulates year-agnostic stable chunks (địa chỉ,
    hotline) that the IN_SEASON+current POST-FETCH hard-drop MUST
    preserve.
    """
    return {
        "text": f"text-for-{chunk_id}",
        "source": "synthetic",
        "chunk_id": chunk_id,
        "id": chunk_id,
        "metadata_json": json.dumps({"data_year": data_year, "is_superseded": False}),
        "_distance": 0.5,
    }


# ---------------------------------------------------------------------------
# Helpers — generator (mirror test_generator_season.py)
# ---------------------------------------------------------------------------


def _make_gen(answer_text: str = "Câu trả lời mặc định từ mô hình.") -> Any:
    """Construct an LLMGenerator with a mocked unified client.

    Bypasses `__init__` to avoid loading the prompt file + LLM clients.
    Mirrors the pattern in `tests/services/test_generator_season.py`.
    """
    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST SYSTEM PROMPT"
    gen.zai_client = None
    gen.groq_client = None
    gen.has_any_provider = True
    gen.gen_model = "test-model"
    gen.gen_fallback_model = "test-fallback"
    gen.unified_fallback_client = None

    fake_unified = MagicMock()
    fake_unified._providers = ["primary"]
    resp = MagicMock()
    resp.content = answer_text
    resp.model = "test-model"
    resp.provider = "test"
    fake_unified.chat = AsyncMock(return_value=resp)
    gen.unified_client = fake_unified

    async def _passthrough_vi(answer, *args, **kwargs):
        return answer

    gen._enforce_vietnamese = _passthrough_vi  # type: ignore[assignment]
    return gen


def _stub_context(gen: Any, year: int, season: Any, has_current_year_chunk: bool) -> None:
    """Monkey-patch the admission-context resolver on the instance.

    Drives season-phase behavior without a live retriever or wall clock.
    """

    def _resolver(chunks):
        return year, season, has_current_year_chunk

    gen._resolve_admission_context = _resolver  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helpers — chunk shapes (mirrors what the retriever would emit)
# ---------------------------------------------------------------------------


def _doc_as_chunk(chunk_id: str, *, data_year: Optional[int], text: str) -> Dict[str, Any]:
    """Shape a retriever-emitted document into the chunk dict that
    `LLMGenerator.generate_answer` expects."""
    return {
        "text": text,
        "summary": f"summary-for-{chunk_id}",
        "metadata": {"data_year": data_year, "info_type": "diem_chuan"},
        "source": "synthetic",
        "chunk_id": chunk_id,
    }


# ---------------------------------------------------------------------------
# The E2E test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_10_year_temporal_contract_same_query_three_phases(monkeypatch):
    """The 10-year contract: the SAME query string yields the correct
    temporal behavior across three calendar phases as the admission
    year rolls over. This is the regression guard that keeps the
    system correct for 10 years.

    Encoding: 3 phases x (retriever row-survival + generator disclaimer).

    If any phase assertion fails, that means a real src bug is leaking
    the wrong year or skipping a hard-drop. REPORT it, do not fix it.
    """
    from services.season import SeasonPhase

    # ---------------- Shared setup ----------------
    # Ensure deterministic clock for any authority-fallback code paths.
    monkeypatch.delenv("APP_ENV", raising=False)

    # The SAME query flows through all three phases.
    assert SAME_QUERY, "SAME_QUERY must be a non-empty string"

    # ============================================================
    # PHASE 1 — Jul 2026, in-season 2026
    # ============================================================
    # authority year = 2026, has_data(2026) = True → IN_SEASON
    # rows {2026, 2025, null-year}
    # contract: drop 2025, keep 2026 + null-year; NO gap disclaimer.
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", str(P1_YEAR))

    p1_rows = [
        _row(CID_CURRENT_2026, data_year=2026),
        _row(CID_PRIOR_2025, data_year=2025),
        _row(CID_STABLE_NULL, data_year=None),
    ]
    p1_retriever = _make_retriever(
        rows=p1_rows, year=P1_YEAR, has_data=True
    )
    p1_result = p1_retriever.retrieve(
        query_vector=_make_vector(),
        top_k=5,
        query=SAME_QUERY,
    )
    p1_chunk_ids = {doc.chunk_id for doc in p1_result.documents}

    # P1 retriever contract
    assert CID_CURRENT_2026 in p1_chunk_ids, (
        f"[P1] IN_SEASON+current must KEEP the current-year chunk; "
        f"got {p1_chunk_ids}"
    )
    assert CID_STABLE_NULL in p1_chunk_ids, (
        f"[P1] IN_SEASON+current must KEEP year-agnostic stable chunks "
        f"(data_year=None); got {p1_chunk_ids}"
    )
    assert CID_PRIOR_2025 not in p1_chunk_ids, (
        f"[P1] IN_SEASON+current must HARD-DROP prior-year chunks "
        f"(data_year=2025); got {p1_chunk_ids}"
    )

    # P1 generator contract: IN_SEASON + has_current_year_chunk=True → no disclaimer.
    p1_model_body = "Năm 2026 điểm chuẩn ngành CNTT HUSC là 19.0 theo TB60."
    p1_gen = _make_gen(answer_text=p1_model_body)
    _stub_context(
        p1_gen,
        year=P1_YEAR,
        season=SeasonPhase.IN_SEASON,
        has_current_year_chunk=True,
    )
    p1_chunks_for_gen = [
        _doc_as_chunk(
            CID_CURRENT_2026,
            data_year=2026,
            text="Năm 2026 điểm chuẩn ngành CNTT là 19.0.",
        ),
        _doc_as_chunk(
            CID_STABLE_NULL,
            data_year=None,
            text="Địa chỉ HUSC: 77 Nguyễn Huệ, Huế.",
        ),
    ]
    p1_out = await p1_gen.generate_answer(
        query=SAME_QUERY,
        chunks=p1_chunks_for_gen,
        confidence=0.9,
    )
    p1_answer = p1_out["answer"]
    assert "chưa được công bố chính thức" not in p1_answer, (
        f"[P1] IN_SEASON answer must NOT contain the gap-disclaimer marker; "
        f"got {p1_answer!r}"
    )
    assert "gap-disclaimer" not in p1_out.get("provider", ""), (
        f"[P1] IN_SEASON must not tag provider with gap-disclaimer; "
        f"got provider={p1_out.get('provider')!r}"
    )

    # ============================================================
    # PHASE 2 — Oct 2026, gap waiting for 2027
    # ============================================================
    # authority year = 2027, has_data(2027) = False → PRE_SEASON_GAP
    # rows {2026, 2025} (no 2027)
    # retriever contract: keep 2025 + 2026 (NOT dropped — show labeled)
    # generator contract: emit canonical GAP_DISCLAIMER + keep prior body.
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", str(P2_YEAR))

    p2_rows = [
        _row(CID_PRIOR_2026, data_year=2026),
        _row(CID_PRIOR_2025, data_year=2025),
    ]
    p2_retriever = _make_retriever(
        rows=p2_rows, year=P2_YEAR, has_data=False
    )
    p2_result = p2_retriever.retrieve(
        query_vector=_make_vector(),
        top_k=5,
        query=SAME_QUERY,
    )
    p2_chunk_ids = {doc.chunk_id for doc in p2_result.documents}

    # P2 retriever contract — CRITICAL invariant: in PRE_SEASON_GAP,
    # prior-year rows MUST survive (we must show them labeled, not
    # drop them — the user has no current-year data yet).
    assert CID_PRIOR_2026 in p2_chunk_ids, (
        f"[P2] PRE_SEASON_GAP must KEEP prior-year 2026 chunk (not drop "
        f"it — no 2027 data exists yet); got {p2_chunk_ids}"
    )
    assert CID_PRIOR_2025 in p2_chunk_ids, (
        f"[P2] PRE_SEASON_GAP must KEEP prior-year 2025 chunk; "
        f"got {p2_chunk_ids}"
    )

    # P2 generator contract: PRE_SEASON_GAP + has prior chunks → emit
    # canonical GAP_DISCLAIMER; keep prior-year body.
    p2_gen = _make_gen(answer_text=P2_PRIOR_BODY)
    _stub_context(
        p2_gen,
        year=P2_YEAR,
        season=SeasonPhase.PRE_SEASON_GAP,
        has_current_year_chunk=False,  # no 2027 chunk in our synthetic set
    )
    p2_chunks_for_gen = [
        _doc_as_chunk(
            CID_PRIOR_2026,
            data_year=2026,
            text="Năm 2026 điểm chuẩn ngành CNTT là 18.5 theo TB60.",
        ),
    ]
    p2_out = await p2_gen.generate_answer(
        query=SAME_QUERY,
        chunks=p2_chunks_for_gen,
        confidence=0.6,
    )
    p2_answer = p2_out["answer"]

    # Canonical disclaimer markers (per the PINNED Canonical strings block
    # in `.omc/plans/temporal_durability_fix_plan.md`).
    assert "chưa được công bố chính thức" in p2_answer, (
        f"[P2] PRE_SEASON_GAP must emit the canonical gap-disclaimer "
        f"marker; got {p2_answer!r}"
    )
    # Year-parametrization: the "chưa công bố" clause must be anchored
    # on the CURRENT year 2027 (NOT on the prior year being shown).
    cut = p2_answer.split(P2_PRIOR_BODY)[0] if P2_PRIOR_BODY in p2_answer else p2_answer
    assert "năm 2027 chưa được công bố" in cut, (
        f"[P2] Disclaimer must anchor the 'chưa công bố' clause on year "
        f"2027 (the current authority year); got disclaimer prefix={cut!r}"
    )
    # The prior-year body MUST be preserved (not suppressed).
    assert P2_PRIOR_BODY in p2_answer, (
        f"[P2] PRE_SEASON_GAP must KEEP the prior-year body (label, not "
        f"suppress); got {p2_answer!r}"
    )
    # Disclaimer must be PREPENDED (before the body), not appended.
    assert p2_answer.index("chưa được công bố chính thức") < p2_answer.index(P2_PRIOR_BODY), (
        f"[P2] Gap disclaimer must be PREPENDED before the prior-year "
        f"body; got {p2_answer!r}"
    )
    # Provider tag.
    assert "gap-disclaimer" in p2_out.get("provider", ""), (
        f"[P2] PRE_SEASON_GAP must tag provider with 'gap-disclaimer'; "
        f"got provider={p2_out.get('provider')!r}"
    )

    # ============================================================
    # PHASE 3 — Apr 2027, in-season 2027 with fresh data
    # ============================================================
    # authority year = 2027, has_data(2027) = True → IN_SEASON
    # rows {2027, 2026, 2025, null-year}
    # contract: drop 2026 + 2025 (now both prior), keep 2027 + null-year
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", str(P3_YEAR))

    p3_rows = [
        _row(CID_CURRENT_2027, data_year=2027),
        _row(CID_PRIOR_2026, data_year=2026),
        _row(CID_PRIOR_2025, data_year=2025),
        _row(CID_STABLE_NULL, data_year=None),
    ]
    p3_retriever = _make_retriever(
        rows=p3_rows, year=P3_YEAR, has_data=True
    )
    p3_result = p3_retriever.retrieve(
        query_vector=_make_vector(),
        top_k=5,
        query=SAME_QUERY,
    )
    p3_chunk_ids = {doc.chunk_id for doc in p3_result.documents}

    # P3 retriever contract — the rollover proof: 2026 was the
    # current-year in Phase 1, but now becomes "prior-year" and is
    # hard-dropped. No code change between phases — only the year +
    # data state changed.
    assert CID_CURRENT_2027 in p3_chunk_ids, (
        f"[P3] IN_SEASON 2027 must KEEP the new current-year chunk; "
        f"got {p3_chunk_ids}"
    )
    assert CID_STABLE_NULL in p3_chunk_ids, (
        f"[P3] IN_SEASON 2027 must KEEP year-agnostic stable chunks; "
        f"got {p3_chunk_ids}"
    )
    assert CID_PRIOR_2026 not in p3_chunk_ids, (
        f"[P3] IN_SEASON 2027 must HARD-DROP 2026 (now a prior year) — "
        f"this is the rollover proof; got {p3_chunk_ids}"
    )
    assert CID_PRIOR_2025 not in p3_chunk_ids, (
        f"[P3] IN_SEASON 2027 must HARD-DROP 2025 (still prior); "
        f"got {p3_chunk_ids}"
    )

    # P3 generator contract: IN_SEASON + has_current_year_chunk=True → no disclaimer.
    p3_model_body = "Năm 2027 điểm chuẩn ngành CNTT HUSC là 20.0 theo TB72."
    p3_gen = _make_gen(answer_text=p3_model_body)
    _stub_context(
        p3_gen,
        year=P3_YEAR,
        season=SeasonPhase.IN_SEASON,
        has_current_year_chunk=True,
    )
    p3_chunks_for_gen = [
        _doc_as_chunk(
            CID_CURRENT_2027,
            data_year=2027,
            text="Năm 2027 điểm chuẩn ngành CNTT là 20.0 theo TB72.",
        ),
        _doc_as_chunk(
            CID_STABLE_NULL,
            data_year=None,
            text="Địa chỉ HUSC: 77 Nguyễn Huệ, Huế.",
        ),
    ]
    p3_out = await p3_gen.generate_answer(
        query=SAME_QUERY,
        chunks=p3_chunks_for_gen,
        confidence=0.95,
    )
    p3_answer = p3_out["answer"]
    assert "chưa được công bố chính thức" not in p3_answer, (
        f"[P3] IN_SEASON 2027 answer must NOT contain the gap-disclaimer "
        f"marker; got {p3_answer!r}"
    )
    assert "gap-disclaimer" not in p3_out.get("provider", ""), (
        f"[P3] IN_SEASON 2027 must not tag provider with gap-disclaimer; "
        f"got provider={p3_out.get('provider')!r}"
    )

    # ============================================================
    # Cross-phase contract assertions
    # ============================================================

    # 1. SAME query string was used in all 3 phases (the test passed
    #    if we got here, but make this explicit for the regression log).
    assert SAME_QUERY == "điểm chuẩn ngành CNTT là bao nhiêu"

    # 2. The KEY cross-phase invariant: P1 dropped 2025 but P2 KEPT
    #    2025. This proves the season gate — we don't hard-drop when
    #    current-year data is absent. If the gate ever leaks, this
    #    assertion will fire.
    assert CID_PRIOR_2025 not in p1_chunk_ids, (
        f"Cross-phase invariant violated: P1 dropped 2025; got {p1_chunk_ids}"
    )
    assert CID_PRIOR_2025 in p2_chunk_ids, (
        f"Cross-phase invariant violated: P2 must KEEP 2025 (no 2027 data "
        f"yet); got {p2_chunk_ids}"
    )
    assert CID_PRIOR_2025 not in p3_chunk_ids, (
        f"Cross-phase invariant violated: P3 must drop 2025; got {p3_chunk_ids}"
    )

    # 3. The KEY rollover proof: the SAME chunk id CID_PRIOR_2026 was
    #    KEPT in P2 (gap) and DROPPED in P3 (in-season 2027). The
    #    ONLY thing that changed is the simulated year + data state.
    assert CID_PRIOR_2026 in p2_chunk_ids, (
        f"Cross-phase invariant violated: P2 must keep 2026 in gap; "
        f"got {p2_chunk_ids}"
    )
    assert CID_PRIOR_2026 not in p3_chunk_ids, (
        f"Cross-phase invariant violated: P3 must drop 2026 (now prior) — "
        f"this is the rollover proof; got {p3_chunk_ids}"
    )

    # 4. null-year (stable) chunks survive in ALL in-season phases
    #    (P1, P3). P2 is gap (not in-season), so stable chunks are
    #    out of scope for this assertion.
    assert CID_STABLE_NULL in p1_chunk_ids, (
        f"Stable-chunk invariant: P1 must keep null-year chunk; "
        f"got {p1_chunk_ids}"
    )
    assert CID_STABLE_NULL in p3_chunk_ids, (
        f"Stable-chunk invariant: P3 must keep null-year chunk; "
        f"got {p3_chunk_ids}"
    )
