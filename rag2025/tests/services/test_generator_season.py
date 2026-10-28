# @spec(S14.4) test_generator_season — season-aware fallback in LLMGenerator
"""TDD: prove the generator degrades gracefully on year-rollover.

Contract pinned by `.omc/plans/temporal_durability_fix_plan.md` (Canonical strings
block + ADR-4 + S14.4):

  * PRE_SEASON_GAP + chunks are all prior-year → PREPEND the canonical
    `GAP_DISCLAIMER` to the model's answer and KEEP the prior-year body
    (do NOT suppress). Assert on stable markers:
        - `"chưa được công bố chính thức"` substring
        - `str(N)` substring (current admission year)
        - prior-year body substring preserved
  * IN_SEASON + a current-year chunk present → NO disclaimer; normal answer.
  * Zero usable chunks + answer-intent → graceful "chưa công bố" fallback
    that mentions year N (NOT a hardcoded "2026") and contains NO fabricated
    HUSC major code (regex `7\\d{6}([A-Z]{2})?`).
  * Year-parametrization: monkeypatch authority to year=2027 → strings
    contain "2027", NEVER "2026".

LLM client is mocked deterministically. The admission-context resolver
(`LLMGenerator._resolve_admission_context`) is stubbed per test so the test
does NOT depend on a live LanceDB or system clock.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# Make `services.*` importable regardless of pytest invocation cwd.
RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


# Canonical major-code shape from `major_code_validator.py:68` — any match in
# the fallback answer is a hallucination.
MAJOR_CODE_RX = re.compile(r"7\d{6}([A-Z]{2})?")


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_gen(answer_text: str = "Câu trả lời mặc định từ mô hình."):
    """Construct an LLMGenerator with a mocked unified client.

    Bypasses `__init__` (which would load the prompt file + LLM clients) to
    keep the test fast and side-effect-free. Mirrors the pattern in
    `tests/services/test_llm_generator_no_fabrication.py`.
    """
    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST SYSTEM PROMPT"
    gen.has_any_provider = True
    gen.gen_model = "test-model"

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


def _stub_context(gen, year: int, season, has_current_year_chunk: bool):
    """Monkey-patch the admission-context resolver on the instance.

    The resolver returns `(year, season, has_current_year_chunk)`. Stubbing
    here lets the test drive season-phase behavior without a live retriever
    or wall-clock.
    """
    def _resolver(chunks):
        return year, season, has_current_year_chunk
    gen._resolve_admission_context = _resolver  # type: ignore[assignment]


def _prior_chunk(year: int = 2025) -> dict:
    """Synthetic prior-year chunk shaped like a retrieval result."""
    return {
        "text": f"Học phí năm {year} ngành CNTT: 14.000.000 VNĐ/năm.",
        "summary": f"Học phí {year}",
        "metadata": {
            "data_year": year,
            "source_url": f"https://tuyensinh.husc.edu.vn/thongbao.php?id=59",
            "source": f"TB59 {year}",
            "info_type": "hoc_phi",
            "notification_id": 59,
        },
    }


def _current_chunk(year: int) -> dict:
    """Synthetic current-year chunk."""
    return {
        "text": f"Học phí năm {year} ngành CNTT: 600.000 VNĐ/tín chỉ theo TB74.",
        "summary": f"Học phí {year}",
        "metadata": {
            "data_year": year,
            "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=74",
            "source": "TB74",
            "info_type": "hoc_phi",
            "notification_id": 74,
        },
    }


# ---------------------------------------------------------------------------
# Case 1 — PRE_SEASON_GAP + prior-year chunks → disclaimer prepended, body kept
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pre_season_gap_prepends_disclaimer_and_preserves_prior_body(monkeypatch):
    """ADR-4 + Canonical strings: when the season is PRE_SEASON_GAP and chunks
    are all prior-year, the answer MUST be prefixed with the canonical
    `GAP_DISCLAIMER` (filled with N, N-1) and the model's body MUST be
    preserved (context is labeled, NOT suppressed)."""
    from services.season import SeasonPhase

    body = "Năm 2025 học phí ngành Công nghệ thông tin là 14.000.000 VNĐ/năm."
    gen = _make_gen(answer_text=body)
    _stub_context(gen, year=2026, season=SeasonPhase.PRE_SEASON_GAP,
                  has_current_year_chunk=False)

    chunks = [_prior_chunk(year=2025)]
    result = await gen.generate_answer(
        query="Học phí 2026 ngành CNTT?",
        chunks=chunks,
        confidence=0.6,
    )
    answer = result["answer"]

    # Stable markers from the PINNED Canonical strings block.
    assert "chưa được công bố chính thức" in answer, (
        f"Gap disclaimer missing the canonical marker: {answer!r}"
    )
    assert "2026" in answer, f"Disclaimer must mention current year N=2026: {answer!r}"
    # Prior body preserved (context labeled, not suppressed).
    assert body in answer, (
        f"Prior-year body must be preserved after disclaimer prepend: {answer!r}"
    )
    # Disclaimer should come BEFORE the body (prepend, not append).
    assert answer.index("chưa được công bố chính thức") < answer.index(body), (
        "Disclaimer must be PREPENDED (appear before the prior-year body)."
    )


# ---------------------------------------------------------------------------
# Case 2 — IN_SEASON + current-year chunk → no disclaimer, normal answer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_in_season_with_current_year_chunk_emits_no_disclaimer(monkeypatch):
    """IN_SEASON + current-year chunk present → normal path. The canonical
    disclaimer marker MUST NOT appear in the answer."""
    from services.season import SeasonPhase

    body = "Học phí năm 2026 ngành CNTT là 600.000 VNĐ/tín chỉ theo TB74."
    gen = _make_gen(answer_text=body)
    _stub_context(gen, year=2026, season=SeasonPhase.IN_SEASON,
                  has_current_year_chunk=True)

    chunks = [_current_chunk(year=2026)]
    result = await gen.generate_answer(
        query="Học phí năm 2026 ngành CNTT bao nhiêu?",
        chunks=chunks,
        confidence=0.9,
    )
    answer = result["answer"]
    assert "chưa được công bố chính thức" not in answer, (
        f"IN_SEASON answer must NOT contain the gap disclaimer: {answer!r}"
    )
    assert answer == body or body in answer, (
        f"IN_SEASON answer should match the model's body verbatim: {answer!r}"
    )
    provider = result.get("provider", "")
    assert "gap-disclaimer" not in provider, (
        f"IN_SEASON path must not tag provider with gap-disclaimer: {provider!r}"
    )


# ---------------------------------------------------------------------------
# Case 3 — empty chunks + answer-intent → graceful year-N fallback, no fabricated codes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_chunks_emits_graceful_year_n_fallback_without_fabrication(monkeypatch):
    """Zero usable chunks (current OR prior) → graceful fallback mentioning
    year N. The fallback string MUST NOT contain any fabricated HUSC major
    code (canonical shape `7\\d{6}([A-Z]{2})?`)."""
    from services.season import SeasonPhase

    # Even if the model returns garbage, the graceful fallback should win.
    gen = _make_gen(answer_text="Năm 2026 dự kiến chỉ tiêu 999 sinh viên ngành 7480201IT.")
    _stub_context(gen, year=2026, season=SeasonPhase.PRE_SEASON_GAP,
                  has_current_year_chunk=False)

    result = await gen.generate_answer(
        query="Học phí 2026 ngành CNTT?",
        chunks=[],  # empty
        confidence=0.0,
    )
    answer = result["answer"]
    assert "2026" in answer, f"Fallback must mention year N=2026: {answer!r}"
    assert not MAJOR_CODE_RX.search(answer), (
        f"Fallback must NOT contain a fabricated HUSC major code: {answer!r}"
    )
    # The fallback must read as a "chưa công bố" graceful disclaimer, not a
    # fabricated answer.
    assert ("chưa được công bố" in answer) or ("chưa công bố" in answer), (
        f"Fallback must use 'chưa công bố' wording: {answer!r}"
    )


# ---------------------------------------------------------------------------
# Case 4 — year-parametrization: 2027 monkeypatch ⇒ strings say "2027", not "2026"
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_year_parametrization_year_2027_propagates_to_gap_disclaimer(monkeypatch):
    """Monkeypatch authority year → 2027. The gap disclaimer MUST contain
    "2027" and MUST NOT contain a literal "2026". Proves no hardcoded year
    leaked into the fallback path."""
    from services.season import SeasonPhase

    body = "Năm 2026 học phí ngành Sinh học là 13.500.000 VNĐ/năm."
    gen = _make_gen(answer_text=body)
    _stub_context(gen, year=2027, season=SeasonPhase.PRE_SEASON_GAP,
                  has_current_year_chunk=False)

    chunks = [_prior_chunk(year=2026)]
    result = await gen.generate_answer(
        query="Học phí 2027 ngành Sinh học?",
        chunks=chunks,
        confidence=0.5,
    )
    answer = result["answer"]
    assert "2027" in answer, (
        f"Disclaimer must mention current year N=2027 after monkeypatch: {answer!r}"
    )
    # Canonical disclaimer shape: "Thông tin tuyển sinh năm {N} chưa được công bố
    # chính thức. Dưới đây là thông tin năm {N-1} (đã cũ, chỉ để tham khảo):"
    # → so the disclaimer prefix CONTAINS BOTH 2027 (the missing year) AND
    # 2026 (the N-1 prior-year label). What proves year-parametrization is
    # that the "chưa được công bố" clause is anchored on 2027, not 2026.
    cut = answer.split(body)[0] if body in answer else answer
    assert "năm 2027 chưa được công bố" in cut, (
        f"Disclaimer must anchor the 'chưa công bố' clause on 2027: {cut!r}"
    )
    # And the N-1 reference must say 2026 (not e.g. 2025 — proves N-1 math).
    assert "năm 2026" in cut, (
        f"Disclaimer must reference N-1=2026 as the prior-year label: {cut!r}"
    )


@pytest.mark.asyncio
async def test_year_parametrization_year_2027_propagates_to_empty_fallback(monkeypatch):
    """Same monkeypatch, empty-chunks path. The fallback must say 2027,
    never 2026."""
    from services.season import SeasonPhase

    gen = _make_gen(answer_text="dummy")
    _stub_context(gen, year=2027, season=SeasonPhase.PRE_SEASON_GAP,
                  has_current_year_chunk=False)

    result = await gen.generate_answer(
        query="Chỉ tiêu tuyển sinh 2027 ngành Toán?",
        chunks=[],
        confidence=0.0,
    )
    answer = result["answer"]
    assert "2027" in answer, f"Empty-fallback must mention 2027: {answer!r}"
    assert "2026" not in answer, (
        f"Empty-fallback must NOT contain literal 2026: {answer!r}"
    )
    assert not MAJOR_CODE_RX.search(answer), (
        f"Empty-fallback must NOT contain a fabricated HUSC major code: {answer!r}"
    )
