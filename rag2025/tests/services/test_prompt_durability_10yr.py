# @spec(S17 Lane C — 10-year prompt durability contract)
r"""TDD: PROVE the HUSC RAG survives 10 years of admission-year rollover and
model swaps with ZERO code/prompt edits — only data ingest + env vars.

This file is the "10-year contract as a test". It is the CI guard that fails
the moment a future developer silently re-introduces a hardcoded admission
year or HUSC major-code literal into:

  * The generation system prompt body (`prompts/generation_system_prompt.txt`),
  * The router prompt CONSTANTS (`_HYDE_SYSTEM`, `_STEP_BACK_SYSTEM`,
    `_CLASSIFY_SYSTEM`, `CONTACT_BLOCK`, `HYDE_REJECT_VAGUE` in
    `src/services/query_router.py`),
  * The temporal authority (must resolve env/clock with NO upper year cap),
  * The generation GAP_DISCLAIMER (must use the dynamically-resolved year).

Why this matters:
  S14 + S16 removed hardcoded majors and years; S17 Lane C *locks the contract*
  so the regression cannot return. If a year-literal or 7-digit major-code
  creeps back in (e.g., somebody pastes "năm 2026" into the step-back prompt
  for a quick fix), the corresponding assertion below MUST fail loudly in CI
  with a precise pointer to the offending source.

Pinned shapes (single source of truth — DO NOT relax):
  * Major-code shape: ``r"7\d{6}([A-Z]{2})?"`` (matches 7480107, 7480107TD,
    7220104, 7440102SC, etc. — pinned by
    `src/services/major_code_validator.py:68` and by
    `tests/services/test_generation_prompt_dehardcoded.py`).
  * Bare admission-year shape: ``r"\b20\d{2}\b"`` (matches 2024, 2025, 2026,
    …, 2099 — the full 10-year-plus window the system must survive).

This test is intentionally strict (zero matches). Educational placeholders
like ``<NĂM>`` / ``<MÃ_NGÀNH>`` / ``20XX`` / ``năm hiện tại`` are the
sanctioned alternatives — none of them trip the regexes above.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


# Make sure ``src.services.*`` and bare ``services.*`` imports both work
# regardless of the pytest invocation cwd. The existing test suite uses
# both styles (e.g., ``test_generator_season.py`` prepends ``rag2025/src``
# and ``test_query_router_autoanswer.py`` prepends ``rag2025``).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RAG_SRC = _REPO_ROOT / "src"
for p in (_REPO_ROOT, _RAG_SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# --------------------------------------------------------------------------
# Pinned regex shapes — DO NOT relax. These are the 10-year contract.
# --------------------------------------------------------------------------

# Canonical HUSC major-code shape: 7 + 6 digits + optional 2-letter suffix.
# Mirrors `src/services/major_code_validator.py` and the S14.6 contract in
# `test_generation_prompt_dehardcoded.py`.
MAJOR_CODE_RX = re.compile(r"7\d{6}([A-Z]{2})?")

# Canonical bare admission-year shape: any 4-digit 20xx year as a standalone
# token. Catches 2024, 2025, 2026, …, 2099 — covers the full 10-year-plus
# window the system must survive without code edits.
YEAR_LITERAL_RX = re.compile(r"\b20\d{2}\b")

# Path to the static generation prompt file (loaded by `LLMGenerator.__init__`).
PROMPT_PATH = _REPO_ROOT / "prompts" / "generation_system_prompt.txt"


# --------------------------------------------------------------------------
# Section 1 — Generation prompt FILE: placeholder + zero year/major literals
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prompt_text() -> str:
    """Read the static prompt file once per module.

    The prompt is the most-edited surface in the system and the highest-risk
    place for a year/major literal to creep back in. We read it directly
    (not the post-``{{YEAR_FACTS}}`` injected version) because the contract
    is about the STATIC body.
    """
    assert PROMPT_PATH.exists(), f"Prompt file missing: {PROMPT_PATH}"
    return PROMPT_PATH.read_text(encoding="utf-8")


def test_prompt_file_contains_year_facts_placeholder(prompt_text: str) -> None:
    """The ``{{YEAR_FACTS}}`` runtime injection point MUST exist exactly once.

    If it is removed/renamed, `LLMGenerator.__init__` silently does nothing
    and the generator falls back to NO year-specific facts (stale-on-rollover
    regression). The exactly-one check rejects accidental duplication that
    would corrupt the briefing block.
    """
    assert "{{YEAR_FACTS}}" in prompt_text, (
        "Prompt file is missing the {{YEAR_FACTS}} placeholder. "
        "Runtime year_facts injection in LLMGenerator.__init__ will be a no-op."
    )
    assert prompt_text.count("{{YEAR_FACTS}}") == 1, (
        "Expected exactly one {{YEAR_FACTS}} placeholder; found "
        f"{prompt_text.count('{{YEAR_FACTS}}')}."
    )


def test_prompt_file_has_zero_major_code_literals(prompt_text: str) -> None:
    """Zero matches of the canonical HUSC major-code shape in the static body.

    Catches: 7480107, 7480107TD, 7440102, 7440102SC, 7510302IC, 7480201VJ,
    7460108, 7220104, 7310301, 7320101, and ANY 7xxxxxx / 7xxxxxxAB form.

    Static major codes must come from retrieved context + ``{{YEAR_FACTS}}``,
    never the prompt — else the model leans on baked codes (a hallucination
    + stale-on-rollover source).
    """
    matches = MAJOR_CODE_RX.findall(prompt_text)
    assert matches == [], (
        "Prompt body still contains hardcoded HUSC major-code literals "
        f"({len(matches)} match(es)). First few: {matches[:5]}. "
        "Strip these — they belong in {{YEAR_FACTS}} or retrieved context."
    )


def test_prompt_file_has_zero_bare_admission_year_literals(prompt_text: str) -> None:
    """Zero 4-digit 20xx year literals in the static body.

    Catches any bare year like 2024 / 2025 / 2026 / … / 2099 anywhere in
    the file (factual claim, example sentence, anything). The prompt must
    be year-agnostic; year facts arrive via ``{{YEAR_FACTS}}`` at runtime.

    Educational placeholders that survive this test: ``<NĂM>``, ``20XX``,
    ``năm hiện tại``, ``năm N``. Use those, not literal years.
    """
    matches = YEAR_LITERAL_RX.findall(prompt_text)
    assert matches == [], (
        "Prompt body still contains bare admission-year literals "
        f"({len(matches)} match(es), unique: {sorted(set(matches))}). "
        "Replace with '<NĂM>' / 'năm hiện tại' / '{{YEAR_FACTS}}' injection."
    )


# --------------------------------------------------------------------------
# Section 2 — Router prompt CONSTANTS: zero year/major literals
# --------------------------------------------------------------------------


# Import after sys.path manipulation so both ``src.services.*`` style works
# under pytest invoked from the repo root or from ``rag2025/``.
from src.services.query_router import (  # noqa: E402
    HYDE_REJECT_VAGUE,
    _CLASSIFY_SYSTEM,
    _HYDE_SYSTEM,
    _STEP_BACK_SYSTEM,
)
from src.services.contact_block import CONTACT_BLOCK  # noqa: E402


# Map of (constant-name, constant-value) tuples for parametrized assertions.
# Adding a new prompt constant later? Append it here — the durability gate
# extends automatically.
_ROUTER_PROMPT_CONSTANTS = [
    ("_HYDE_SYSTEM", _HYDE_SYSTEM),
    ("_STEP_BACK_SYSTEM", _STEP_BACK_SYSTEM),
    ("_CLASSIFY_SYSTEM", _CLASSIFY_SYSTEM),
    ("CONTACT_BLOCK", CONTACT_BLOCK),
    ("HYDE_REJECT_VAGUE", HYDE_REJECT_VAGUE),
]


@pytest.mark.parametrize("name,value", _ROUTER_PROMPT_CONSTANTS)
def test_router_prompt_constant_has_zero_major_code_literals(name: str, value: str) -> None:
    """Each router prompt CONSTANT must have ZERO HUSC major-code literals.

    Router prompts are sent to the classify/HyDE/step-back LLM on every
    query. A baked major code (e.g., ``7480101``) tells the model that one
    specific major is somehow canonical, biases the step-back/HyDE generation,
    and survives any data-ingest refresh (silent stale-on-rollover).
    """
    matches = MAJOR_CODE_RX.findall(value)
    assert matches == [], (
        f"Router constant {name!r} contains a HUSC major-code literal "
        f"({len(matches)} match(es), e.g. {matches[:3]}). "
        "Use '<MÃ_NGÀNH>' / '7xxxxxx' placeholder instead — the constant "
        "must be year-agnostic AND major-agnostic so it survives 10+ years "
        "and any catalogue change."
    )


@pytest.mark.parametrize("name,value", _ROUTER_PROMPT_CONSTANTS)
def test_router_prompt_constant_has_zero_bare_admission_year_literals(
    name: str, value: str
) -> None:
    """Each router prompt CONSTANT must have ZERO bare 4-digit admission years.

    CONTACT_BLOCK was already year-agnostic in S16; this test KEEPS it so.
    HYDE_REJECT_VAGUE + _STEP_BACK_SYSTEM + _CLASSIFY_SYSTEM previously
    carried example years like ``2026`` / ``2025`` baked into example
    sentences — S17 Lane C replaced them with the ``<NĂM>`` / ``năm hiện
    tại`` placeholder pattern. This test prevents the regression.
    """
    matches = YEAR_LITERAL_RX.findall(value)
    assert matches == [], (
        f"Router constant {name!r} contains a bare admission-year literal "
        f"({len(matches)} match(es), unique: {sorted(set(matches))}). "
        "Replace with '<NĂM>' / 'năm hiện tại' / 'năm N' so the constant "
        "survives 10+ years of rollover without edits."
    )


# --------------------------------------------------------------------------
# Section 3 — temporal_authority: env override + NO upper cap
# --------------------------------------------------------------------------


def test_temporal_authority_resolves_env_override_2030(monkeypatch) -> None:
    """``CURRENT_ADMISSION_YEAR=2030`` must resolve to 2030.

    This proves env-driven year is the source of truth. The "2030" choice
    is past the current admission year and inside the 10-year window — a
    realistic mid-window rollover scenario.
    """
    from src.services.temporal_authority import get_current_admission_year

    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2030")
    monkeypatch.delenv("APP_ENV", raising=False)
    assert get_current_admission_year() == 2030, (
        "Env override CURRENT_ADMISSION_YEAR=2030 must resolve to 2030. "
        "If this fails, the year authority has a hidden cap or precedence "
        "regression."
    )


def test_temporal_authority_has_no_upper_year_cap(monkeypatch) -> None:
    """``CURRENT_ADMISSION_YEAR=2099`` must resolve to 2099 — NO upper cap.

    S14 deliberately removed any upper-year ceiling so the system survives
    10+ years (and beyond). A regression that caps at e.g. 2030 or 2035
    would silently break in year N+1. Pin the contract at 2099 — the
    extreme end of the 21st century — so any cap regression is caught
    immediately, not in production.
    """
    from src.services.temporal_authority import get_current_admission_year

    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2099")
    monkeypatch.delenv("APP_ENV", raising=False)
    assert get_current_admission_year() == 2099, (
        "Env override CURRENT_ADMISSION_YEAR=2099 must resolve to 2099. "
        "If this fails, somebody added an upper-year cap — REMOVE it. "
        "The system must survive 10+ years with zero code edits."
    )


def test_temporal_authority_env_takes_precedence_over_clock(monkeypatch) -> None:
    """Env override wins over the calendar fallback (sanity guard).

    A regression where the calendar fallback shadows the env value would
    break prod deploys that pin the year via env. This is the contract
    `temporal_authority.py` advertises in its module docstring.
    """
    from datetime import datetime
    from unittest.mock import patch

    from src.services.temporal_authority import get_current_admission_year

    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2035")
    monkeypatch.delenv("APP_ENV", raising=False)
    # Clock says a much earlier year — env must still win.
    fake_now = datetime(2027, 10, 1)
    with patch("src.services.temporal_authority._now", return_value=fake_now):
        assert get_current_admission_year() == 2035, (
            "Env override must beat the calendar fallback. If this fails, "
            "the precedence order in get_current_admission_year() regressed."
        )


def test_year_normalizer_explicit_year_window_grows_with_current_year(monkeypatch):
    """A future explicit year (current_year + 5) inside the dynamic window must
    prevent relative-phrase substitution.

    Regression guard: the old regex ``20[2-3]\\d`` capped recognition at 2039 and
    would have rewritten a 2040+ phrase. The window must be relative to
    current_year, not a fixed literal ceiling.
    """
    from datetime import datetime
    from unittest.mock import patch

    from src.services import temporal_authority
    from src.services import year_normalizer

    # Drive current_year to 2041 so current_year+5 = 2046 is inside the window.
    fake_now = datetime(2041, 3, 15)
    with patch.object(temporal_authority, "_now", return_value=fake_now):
        with patch.object(year_normalizer, "_now", return_value=fake_now):
            current_year = temporal_authority.get_current_admission_year()
            future_year = current_year + 5
            query = f"Năm nay học phí năm {future_year}?"
            out, rep = year_normalizer.normalize_relative_year(query, current_year)
            assert out == query
            assert rep == []


def test_year_normalizer_explicit_year_outside_window_is_substituted(monkeypatch):
    """A year beyond current_year+6 is too far in the future; relative phrases
    should still be substituted because the explicit year is outside the valid
    admission-data window.
    """
    from datetime import datetime
    from unittest.mock import patch

    from src.services import temporal_authority
    from src.services import year_normalizer

    fake_now = datetime(2026, 3, 15)
    with patch.object(temporal_authority, "_now", return_value=fake_now):
        with patch.object(year_normalizer, "_now", return_value=fake_now):
            current_year = temporal_authority.get_current_admission_year()
            far_year = current_year + 8
            query = f"Năm nay học phí năm {far_year}?"
            out, rep = year_normalizer.normalize_relative_year(query, current_year)
            # The explicit far-year is ignored; relative "năm nay" is rewritten.
            assert out != query
            assert "năm 2026" in out
            assert len(rep) == 1
            assert rep[0].phrase.lower() == "năm nay"


# --------------------------------------------------------------------------
# Section 4 — Generation GAP_DISCLAIMER uses the dynamic year (2099 probe)
# --------------------------------------------------------------------------


def _make_gen_stub(answer_text: str = "Dummy model answer.") -> "object":
    """Construct an LLMGenerator with a mocked unified client.

    Mirrors the pattern in ``tests/services/test_generator_season.py`` —
    bypass ``__init__`` (which would load the prompt file + LLM clients)
    to keep the test fast and side-effect-free.
    """
    from services.llm_generator import LLMGenerator

    gen = LLMGenerator.__new__(LLMGenerator)
    gen.generation_system_prompt = "TEST SYSTEM PROMPT"
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


def _stub_admission_context(gen, year: int, season, has_current_year_chunk: bool):
    """Monkey-patch the admission-context resolver onto the instance.

    The real resolver imports a LanceDB retriever; stubbing here lets the
    test drive season-phase behaviour deterministically without I/O.
    """
    def _resolver(chunks):
        return year, season, has_current_year_chunk
    gen._resolve_admission_context = _resolver  # type: ignore[assignment]


def _prior_year_chunk(year: int) -> dict:
    """Synthetic prior-year chunk shaped like a retrieval result."""
    return {
        "text": f"Học phí năm {year} ngành CNTT: 14.000.000 VNĐ/năm.",
        "summary": f"Học phí {year}",
        "metadata": {
            "data_year": year,
            "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=59",
            "source": f"TB59 {year}",
            "info_type": "hoc_phi",
            "notification_id": 59,
        },
    }


@pytest.mark.asyncio
async def test_gap_disclaimer_propagates_year_2099_not_hardcoded(monkeypatch) -> None:
    """Monkeypatch authority to year=2099, force PRE_SEASON_GAP, supply a
    prior-year chunk → the GAP_DISCLAIMER MUST contain "2099" and MUST NOT
    contain a hardcoded "2026" anywhere in the prepended disclaimer prefix.

    Why 2099: it's the extreme of the 10-year-plus window. If anything in
    the season-aware fallback is hardcoded to the current admission year,
    asserting on 2099 makes the regression unmissable.

    Reuses the construction pattern from
    ``tests/services/test_generator_season.py`` (`_make_gen`, `_stub_context`,
    `_prior_chunk`) — same shapes, same stubs, just a more extreme year.
    """
    from services.season import SeasonPhase

    body = "Năm 2098 học phí ngành Sinh học là 13.500.000 VNĐ/năm."
    gen = _make_gen_stub(answer_text=body)
    _stub_admission_context(
        gen, year=2099, season=SeasonPhase.PRE_SEASON_GAP,
        has_current_year_chunk=False,
    )

    chunks = [_prior_year_chunk(year=2098)]
    result = await gen.generate_answer(
        query="Học phí 2099 ngành Sinh học?",
        chunks=chunks,
        confidence=0.5,
    )
    answer = result["answer"]

    # Canonical GAP_DISCLAIMER shape from llm_generator.py:604-606:
    #   "Thông tin tuyển sinh năm {N} chưa được công bố chính thức.
    #    Dưới đây là thông tin năm {N-1} (đã cũ, chỉ để tham khảo):"
    # → so the disclaimer prefix CONTAINS BOTH 2099 (the missing year) AND
    # 2098 (the N-1 prior-year label). What proves year-parametrization is
    # that the "chưa được công bố" clause is anchored on 2099.
    assert "2099" in answer, (
        f"GAP_DISCLAIMER must mention monkeypatched year 2099, got: {answer!r}"
    )
    disclaimer_prefix = answer.split(body)[0] if body in answer else answer
    assert "năm 2099 chưa được công bố" in disclaimer_prefix, (
        "GAP_DISCLAIMER must anchor the 'chưa công bố' clause on monkeypatched "
        f"year 2099. Got prefix: {disclaimer_prefix!r}"
    )
    # The static prompt year (2026) must NOT leak into the dynamic disclaimer.
    # We allow "2098" (the N-1 label, which IS dynamically derived from N=2099)
    # but explicitly forbid "2026" anywhere in the prefix.
    assert "2026" not in disclaimer_prefix, (
        "GAP_DISCLAIMER prefix must NOT contain a hardcoded '2026'. Got: "
        f"{disclaimer_prefix!r}"
    )
    # And the disclaimer must NOT fabricate a HUSC major code in its
    # year-parametrized prefix.
    assert not MAJOR_CODE_RX.search(disclaimer_prefix), (
        "GAP_DISCLAIMER must NOT fabricate a HUSC major code in its prefix: "
        f"{disclaimer_prefix!r}"
    )


@pytest.mark.asyncio
async def test_empty_chunks_fallback_uses_year_2099_not_hardcoded(monkeypatch) -> None:
    """Same monkeypatch (year=2099), but the EMPTY-chunks fallback path.

    With zero usable chunks + answer-intent, llm_generator emits the
    season-fallback "chưa được công bố chính thức" string parametrized
    on the resolved current_year. The string MUST contain "2099" and
    MUST NOT contain any literal current-era year (2026) or fabricated
    major code.
    """
    from services.season import SeasonPhase

    gen = _make_gen_stub(answer_text="Mô hình trả ra rác, fallback phải thắng.")
    _stub_admission_context(
        gen, year=2099, season=SeasonPhase.PRE_SEASON_GAP,
        has_current_year_chunk=False,
    )

    result = await gen.generate_answer(
        query="Chỉ tiêu tuyển sinh 2099 ngành Toán?",
        chunks=[],
        confidence=0.0,
    )
    answer = result["answer"]
    assert "2099" in answer, (
        f"Empty-chunks fallback must mention monkeypatched year 2099: {answer!r}"
    )
    assert "2026" not in answer, (
        f"Empty-chunks fallback must NOT contain literal 2026: {answer!r}"
    )
    assert not MAJOR_CODE_RX.search(answer), (
        f"Empty-chunks fallback must NOT fabricate a HUSC major code: {answer!r}"
    )
    assert ("chưa được công bố" in answer) or ("chưa công bố" in answer), (
        f"Empty-chunks fallback must use 'chưa công bố' wording: {answer!r}"
    )
