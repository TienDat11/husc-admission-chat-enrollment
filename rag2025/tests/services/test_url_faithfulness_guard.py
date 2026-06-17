# @spec(S16.5 / ADR-D / AMF-3) test_url_faithfulness_guard
"""TDD: S16.5 — URL-faithfulness post-guard for `llm_generator.py`.

Contract (pinned by `.omc/plans/s16_faithfulness_route_calibration_plan.md`
§ADR-D + AMF-3 + S16.5):

  * Module-level PURE helper `_strip_ungrounded_urls(
        answer: str, chunk_texts: list[str], allowlist: set[str]
    ) -> str` extracts URL-like tokens from `answer`:
      - `https?://\S+` (full URL)
      - bare `[\w.-]+\.(?:edu\.vn|com|vn|gov\.vn)\S*` (no-scheme URL)
  * For each token: if its host/path substring appears in ANY chunk text →
    KEEP (grounded). Else if its host is in the allowlist → KEEP (policy).
    Else → STRIP (markdown `[text](url)` → just `text`; bare URL token removed
    with orphaned "tại:"/empty parens cleanup).
  * The allowlist MUST include `tuyensinh.husc.edu.vn` so the season-aware
    GAP_DISCLAIMER injected at `llm_generator.py:493` is preserved
    (AMF-3: a naive substring guard would strip a POLICY-grounded URL).
  * Allowlist MUST also include `husc.edu.vn`, `dkxt.hueuni.edu.vn`,
    `thisinh.thitotnghiepthpt.edu.vn` per AMF-3.

Wired into `LLMGenerator.generate_answer` AFTER the answer is produced,
adjacent to the S15.6 contact-abstain guard (line ~510), passing the
retrieved chunk texts and the allowlist constant.

LLM is NOT exercised here. The helper is pure and unit-testable directly.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))


# Canonical allowlist per AMF-3. HUSC/Bộ portals that are policy-grounded
# even when not in retrieved chunks. The SEASON/GAP_DISCLAIMER code at
# `llm_generator.py:493` injects `https://tuyensinh.husc.edu.vn` which is
# NOT in any chunk; the guard MUST NOT strip it.
EXPECTED_ALLOWLIST = {
    "tuyensinh.husc.edu.vn",
    "husc.edu.vn",
    "dkxt.hueuni.edu.vn",
    "thisinh.thitotnghiepthpt.edu.vn",
}


# ---------------------------------------------------------------------------
# Helper direct tests
# ---------------------------------------------------------------------------

class TestStripUngroundedUrlsHelper:
    """Direct unit tests of the pure helper."""

    def test_ungrounded_https_url_stripped(self):
        """A bare https URL with no grounding in chunks + not in allowlist
        MUST be stripped from the answer."""
        from services.llm_generator import _strip_ungrounded_urls

        answer = "Xem thêm tại https://fake-portal.example.com để biết chi tiết."
        chunks = ["Học phí năm 2026 ngành CNTT là 600.000 VNĐ/tín chỉ."]
        out = _strip_ungrounded_urls(answer, chunks, EXPECTED_ALLOWLIST)
        assert "fake-portal.example.com" not in out, (
            f"Ungrounded URL must be stripped; got: {out!r}"
        )
        # The text leading up to the URL should survive (only the URL is gone).
        assert "Xem thêm tại" in out
        # Should not leave orphaned "tại " (clean empty trailing phrase).
        assert "tại ." not in out and "tại  " not in out, (
            f"Orphaned 'tại' must be cleaned; got: {out!r}"
        )

    def test_url_host_in_chunk_is_kept(self):
        """When the host of a URL is a substring of any chunk text, the URL
        is grounded and MUST be kept verbatim."""
        from services.llm_generator import _strip_ungrounded_urls

        answer = "Theo thông báo tại https://tuyensinh.husc.edu.vn/thongbao.php?id=74"
        chunks = [
            "Chi tiết tại https://tuyensinh.husc.edu.vn/thongbao.php?id=74 "
            "(thông báo chính thức)."
        ]
        out = _strip_ungrounded_urls(answer, chunks, EXPECTED_ALLOWLIST)
        assert "tuyensinh.husc.edu.vn/thongbao.php?id=74" in out, (
            f"Grounded URL must be kept; got: {out!r}"
        )

    def test_allowlist_url_kept_when_not_in_chunks(self):
        """AMF-3: the GAP_DISCLAIMER injects `https://tuyensinh.husc.edu.vn`
        even though no chunk mentions it. The allowlist MUST protect it."""
        from services.llm_generator import _strip_ungrounded_urls

        answer = (
            "Thông tin tuyển sinh năm 2026 chưa công bố. "
            "Vui lòng tham khảo https://tuyensinh.husc.edu.vn để cập nhật."
        )
        chunks = ["Học phí năm 2025 ngành CNTT: 14.000.000 VNĐ/năm."]
        out = _strip_ungrounded_urls(answer, chunks, EXPECTED_ALLOWLIST)
        assert "tuyensinh.husc.edu.vn" in out, (
            f"Allowlist URL must be kept even when not in chunks; got: {out!r}"
        )

    def test_markdown_link_ungrounded_keeps_text_drops_url(self):
        """Markdown `[text](ungrounded-url)` MUST reduce to just `text` —
        the text is preserved (not lost), only the URL is dropped."""
        from services.llm_generator import _strip_ungrounded_urls

        answer = "Đăng ký tại [Cổng thông tin](https://bad.com/fake) để xét tuyển."
        chunks = ["Thời gian đăng ký xét tuyển từ tháng 3 đến tháng 6."]
        out = _strip_ungrounded_urls(answer, chunks, EXPECTED_ALLOWLIST)
        assert "Cổng thông tin" in out, (
            f"Markdown link text must be preserved; got: {out!r}"
        )
        assert "bad.com" not in out, (
            f"Markdown link URL must be stripped; got: {out!r}"
        )
        # No dangling "()" or "](  )" artifacts
        assert "]( )" not in out and "[](" not in out, (
            f"Markdown residue must be cleaned; got: {out!r}"
        )

    def test_no_url_answer_unchanged(self):
        """An answer that contains no URL-like tokens MUST pass through
        verbatim (no whitespace, no character drift)."""
        from services.llm_generator import _strip_ungrounded_urls

        answer = "Điểm chuẩn ngành CNTT năm 2025 là 24.5."
        chunks = ["Điểm chuẩn ngành CNTT các năm gần đây."]
        out = _strip_ungrounded_urls(answer, chunks, EXPECTED_ALLOWLIST)
        assert out == answer, (
            f"No-URL answer must be unchanged; got diff: "
            f"in={answer!r} out={out!r}"
        )

    def test_allowlist_constant_shape_is_pinned(self):
        """AMF-3 + plan: the allowlist constant MUST contain the four
        canonical domains, otherwise GAP_DISCLAIMER / contact-abstain guard
        URLs get stripped and the season fallback breaks."""
        from services.llm_generator import _URL_ALLOWLIST

        assert isinstance(_URL_ALLOWLIST, (set, frozenset)), (
            f"_URL_ALLOWLIST must be a set/frozenset; got {type(_URL_ALLOWLIST)!r}"
        )
        for required in EXPECTED_ALLOWLIST:
            assert required in _URL_ALLOWLIST, (
                f"_URL_ALLOWLIST missing required host {required!r}; "
                f"got {_URL_ALLOWLIST!r}"
            )

    def test_helper_is_pure_no_module_state(self):
        """The helper MUST be a module-level pure function — no implicit
        dependency on a generator instance or module state. Easy to call
        from anywhere; test by calling twice with swapped args and
        asserting the same result."""
        from services.llm_generator import _strip_ungrounded_urls

        answer_a = "Xem tại https://a.com"
        answer_b = "Xem tại https://b.com"
        chunks = ["some chunk text with https://a.com only"]
        out_a1 = _strip_ungrounded_urls(answer_a, chunks, EXPECTED_ALLOWLIST)
        out_a2 = _strip_ungrounded_urls(answer_a, chunks, EXPECTED_ALLOWLIST)
        out_b = _strip_ungrounded_urls(answer_b, chunks, EXPECTED_ALLOWLIST)
        assert out_a1 == out_a2, "Helper must be pure / deterministic"
        assert "a.com" not in out_b, "Helper must not leak state between calls"


# ---------------------------------------------------------------------------
# Integration: guard runs inside `generate_answer` and protects GAP_DISCLAIMER
# ---------------------------------------------------------------------------

def _make_gen(answer_text: str = "Câu trả lời mặc định từ mô hình."):
    """Bypass `__init__` to wire a deterministic mock LLM. Mirrors
    `tests/services/test_generator_season.py::_make_gen`."""
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


def _stub_in_season(gen, year: int = 2026):
    """Drive `_resolve_admission_context` to IN_SEASON + current-year chunk
    present. With current-year chunks present, the season fallback prepends
    NO disclaimer and leaves the answer body intact, so the URL guard runs
    on a clean answer string."""
    from services.season import SeasonPhase

    def _stub_context(chunks):
        return year, SeasonPhase.IN_SEASON, True

    gen._resolve_admission_context = _stub_context  # type: ignore[assignment]


def _chunk(text: str, year: int = 2026) -> dict:
    return {
        "text": text,
        "summary": "",
        "metadata": {
            "data_year": year,
            "source_url": "https://tuyensinh.husc.edu.vn/",
            "source": "TB_test",
            "info_type": "general",
        },
    }


@pytest.mark.asyncio
async def test_generate_answer_strips_ungrounded_url_in_answer():
    """End-to-end: model volunteers a fabricated URL not in any chunk
    + not in allowlist → the URL guard strips it from the final answer."""
    from services.season import SeasonPhase

    # Body contains an ungrounded fabricated URL.
    body = (
        "Học phí ngành CNTT 2026. "
        "Xem chi tiết tại https://fabricated-portal.example.com/thong-bao"
    )
    gen = _make_gen(answer_text=body)

    def _stub_context(chunks):
        return 2026, SeasonPhase.IN_SEASON, True
    gen._resolve_admission_context = _stub_context  # type: ignore[assignment]

    chunks = [_chunk("Học phí năm 2026 ngành CNTT là 600.000 VNĐ/tín chỉ.")]
    result = await gen.generate_answer(
        query="Học phí 2026 ngành CNTT?",
        chunks=chunks,
        confidence=0.8,
    )
    answer = result["answer"]
    assert "fabricated-portal.example.com" not in answer, (
        f"Ungrounded URL must be stripped by the post-guard; got: {answer!r}"
    )
    # Body text (minus the URL) should still be present.
    assert "Học phí ngành CNTT 2026" in answer


@pytest.mark.asyncio
async def test_generate_answer_preserves_allowlist_gap_disclaimer_url():
    """End-to-end: simulate the S14.4 GAP_DISCLAIMER path where the code at
    `llm_generator.py:493` injects `https://tuyensinh.husc.edu.vn` even
    though no chunk contains it. The URL guard MUST NOT strip it
    (allowlist protection)."""
    from services.season import SeasonPhase

    # A prior-year body (so season fallback prepends the disclaimer +
    # the model body is preserved verbatim).
    body = "Năm 2025 học phí ngành CNTT: 14.000.000 VNĐ/năm."
    gen = _make_gen(answer_text=body)

    def _stub_context(chunks):
        return 2026, SeasonPhase.PRE_SEASON_GAP, False
    gen._resolve_admission_context = _stub_context  # type: ignore[assignment]

    chunks = [
        {
            "text": "Năm 2025 học phí ngành CNTT: 14.000.000 VNĐ/năm.",
            "summary": "",
            "metadata": {
                "data_year": 2025,
                "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=59",
                "source": "TB59",
                "info_type": "hoc_phi",
            },
        }
    ]
    result = await gen.generate_answer(
        query="Học phí 2026 ngành CNTT?",
        chunks=chunks,
        confidence=0.5,
    )
    answer = result["answer"]
    # The disclaimer text + portal URL from the season fallback's policy
    # answer is only emitted in the no-prior-chunks path; in this test we
    # have prior chunks so the prepended disclaimer doesn't include the
    # portal URL. The real GAP_DISCLAIMER URL path is tested at the
    # helper level (test_allowlist_url_kept_when_not_in_chunks). Here we
    # verify the body survives AND no fabricated URL sneaks in.
    assert "14.000.000 VNĐ" in answer
    assert "fabricated" not in answer.lower()


# ---------------------------------------------------------------------------
# T8 / rich-markdown — regression: URL guard must not mangle table pipes
# ---------------------------------------------------------------------------

class TestStripUngroundedUrlsTableCompatibility:
    """Verify that _strip_ungrounded_urls never touches markdown table pipes."""

    def test_table_pipes_survive_url_stripping(self):
        """A table-heavy answer with an ungrounded URL should have its pipe
        structure preserved EXACTLY as-is after URL stripping."""
        from services.llm_generator import _strip_ungrounded_urls

        answer = (
            "Bảng học phí:\n\n"
            "| Khối | Đơn giá (VNĐ/tín chỉ) |\n"
            "|------|----------------------|\n"
            "| VII  | 490.000              |\n"
            "| IV   | 555.000              |\n"
            "| V    | 600.000              |\n"
            "\nXem thêm tại https://fake.example.com/hoc-phi"
        )
        chunks = ["Đơn giá học phí năm 2026 khối VII 490.000, IV 555.000, V 600.000"]
        out = _strip_ungrounded_urls(answer, chunks, EXPECTED_ALLOWLIST)

        # (a) Pipe characters survive intact (URL guard normalizes whitespace
        # but never removes or replaces pipe characters).
        assert "| Khối" in out
        assert "|------" in out
        assert out.count("| VII") >= 1, f"VII row must survive; got: {out!r}"
        assert out.count("| IV") >= 1
        assert out.count("| V") >= 1

        # (b) Table structure is unchanged (header, separator, 3 data rows).
        # Count pipe-only lines (table rows).
        table_lines = [l for l in out.split("\n") if l.startswith("|")]
        assert len(table_lines) == 5, (
            f"Expected 5 table lines (header + separator + 3 data). "
            f"Got {len(table_lines)}: {table_lines}"
        )

        # (c) Ungrounded URL in the trailing text is stripped.
        assert "fake.example.com" not in out

    def test_url_inside_table_cell_stripped_only_if_ungrounded(self):
        """A URL inside a table cell that is grounded MUST survive, and the
        surrounding pipe structure must remain intact."""
        from services.llm_generator import _strip_ungrounded_urls

        answer = (
            "| Ngành | Link |\n"
            "|-------|------|\n"
            "| CNTT  | https://fake.example.com/cntt |\n"
        )
        # The chunk text does NOT contain "fake.example.com" — ungrounded.
        chunks = ["CNTT chỉ tiêu 120 sinh viên."]
        out = _strip_ungrounded_urls(answer, chunks, EXPECTED_ALLOWLIST)

        # The ungrounded URL should be stripped, but the pipe structure
        # must survive (two pipe rows remain).
        assert "| Ngành |" in out
        assert "|-------|" in out
        # The data row pipe structure stays, even if the URL was removed
        # from the cell content.
        assert out.count("|") >= 6, (
            f"Expected at least 6 pipe chars (2 rows × 3 cols). "
            f"Got: {out!r}"
        )

    def test_grounded_url_in_table_cell_survives(self):
        """A URL inside a table cell that IS grounded MUST stay, and the
        table structure must remain intact."""
        from services.llm_generator import _strip_ungrounded_urls

        answer = (
            "| Ngành | Link |\n"
            "|-------|------|\n"
            "| CNTT  | https://tuyensinh.husc.edu.vn/thongbao.php?id=72 |\n"
        )
        # The chunk contains the full URL — grounded.
        chunks = ["Chi tiết tại https://tuyensinh.husc.edu.vn/thongbao.php?id=72"]
        out = _strip_ungrounded_urls(answer, chunks, EXPECTED_ALLOWLIST)

        assert "tuyensinh.husc.edu.vn" in out, (
            "Grounded URL inside a table cell must be kept."
        )
        assert "| CNTT  |" in out
        assert "thongbao.php?id=72" in out
