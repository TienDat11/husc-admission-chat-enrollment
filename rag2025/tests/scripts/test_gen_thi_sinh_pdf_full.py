"""Regression tests for gen_thi_sinh_pdf.py — guarantees LLM answers are
NOT silently truncated when rendered into the PDF report.

Background
----------
The script previously called ``md_to_rl(answer, <int>)`` with a positive
length cap (300/400/200) on each LLM answer, so the rendered PDF showed
only a prefix of the model's reply. The user reported "không show ra toàn
bộ trả lời của LLMs". The JSON results retain the full text — only the
PDF call sites caused the truncation. These tests lock that fix in.

What we verify
--------------
1. ``md_to_rl(long_text)`` with NO length cap preserves the full visible
   length (no silent 300-char cut-off).
2. Bold markdown (``**...**``) still converts to ``<b>...</b>`` so
   ReportLab inline styling keeps working.
3. XML metacharacters (``&``, ``<``, ``>``) are still escaped before any
   tag injection — otherwise ReportLab Paragraph would choke on raw
   answers that contain HTML-ish text.
4. Source-code regression: no ``md_to_rl(r.get("answer", ...), <int>)``
   call survives in the script. If somebody re-adds a numeric cap on an
   answer call site, this test fails loudly.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("reportlab")

# --------------------------------------------------------------------------- #
# Locate the script. Tests live at rag2025/tests/scripts/, the script at
# rag2025/scripts/. Two ``parent`` hops land on the rag2025 package root.
# --------------------------------------------------------------------------- #
SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "gen_thi_sinh_pdf.py"
)


@pytest.fixture(scope="module")
def gen_module():
    """Load gen_thi_sinh_pdf.py as a module so its ``md_to_rl`` can be
    exercised directly.

    The script runs top-level code at import time (reads the 86Q results
    JSON and registers a Vietnamese TrueType font). Both side effects are
    safe in this repo: the JSON ships with the project under
    ``rag2025/results/`` and the font registration falls back to
    Helvetica if Windows fonts are unavailable, so this works on CI too.
    """
    assert SCRIPT_PATH.exists(), f"Source script missing: {SCRIPT_PATH}"
    spec = importlib.util.spec_from_file_location(
        "gen_thi_sinh_pdf_under_test", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# 1. Length is preserved when no limit is passed.
# --------------------------------------------------------------------------- #
def test_md_to_rl_no_limit_preserves_full_length(gen_module):
    """A ~1500-char Vietnamese paragraph (no emoji, no markdown) must
    survive ``md_to_rl`` essentially intact — proving the old 300-char
    cap has been removed from the call-site contract.
    """
    base = (
        "Trường Đại học Khoa học Huế tuyển sinh năm 2026 theo nhiều phương "
        "thức xét tuyển khác nhau, bao gồm xét điểm thi tốt nghiệp THPT, "
        "xét học bạ THPT, xét tuyển thẳng và ưu tiên xét tuyển theo quy "
        "chế của Bộ Giáo dục và Đào tạo. Thí sinh có thể nộp hồ sơ trực "
        "tuyến qua cổng thông tin tuyển sinh chính thức của nhà trường. "
    )
    long_text = (base * 5)[:1500]
    assert len(long_text) == 1500, "fixture text must be exactly 1500 chars"

    result = gen_module.md_to_rl(long_text)

    # No emoji / markdown in the fixture, so md_to_rl should only collapse
    # whitespace runs — visible length must NOT be capped at 300.
    assert len(result) >= 1400, (
        f"md_to_rl truncated a 1500-char answer to {len(result)} chars — "
        "the answer-call cap has regressed."
    )


# --------------------------------------------------------------------------- #
# 2. Bold markdown still converts to ReportLab inline tags.
# --------------------------------------------------------------------------- #
def test_md_to_rl_bold_conversion(gen_module):
    """``**đậm**`` must become ``<b>đậm</b>`` — Paragraph relies on this
    inline tag for bold rendering.
    """
    result = gen_module.md_to_rl("**đậm** thường")
    assert "<b>đậm</b>" in result, (
        f"Bold markdown conversion broken — got: {result!r}"
    )


# --------------------------------------------------------------------------- #
# 3. XML metachars are still escaped before tag injection.
# --------------------------------------------------------------------------- #
def test_md_to_rl_xml_escape(gen_module):
    """``&``, ``<``, ``>`` must be entity-escaped so ReportLab's
    Paragraph parser does not misinterpret raw answer text as markup.
    """
    result = gen_module.md_to_rl("a & b < c > d")
    assert result == "a &amp; b &lt; c &gt; d", (
        f"XML escape broken — got: {result!r}"
    )


# --------------------------------------------------------------------------- #
# 4. Source-code regression: no positive int cap on answer call sites.
# --------------------------------------------------------------------------- #
def test_no_answer_truncation_call_sites_in_source():
    """Scan the script source. There must be NO call shaped like
    ``md_to_rl(r.get("answer"...), <positive int>)`` left. The fix is
    that every answer rendering call drops its numeric 2nd argument; if
    someone re-adds one (300 / 400 / 200, etc.), this test fails.
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")

    # Match md_to_rl(...) where the 1st arg is r.get("answer", ...) (or
    # r.get('answer', ...)) AND a numeric 2nd arg follows.
    pattern = re.compile(
        r"""md_to_rl\(
            \s*r\.get\(\s*["']answer["'][^)]*\)   # 1st arg: r.get("answer", ...)
            \s*,\s*\d+                            # 2nd arg: a bare integer
            \s*\)""",
        re.VERBOSE | re.DOTALL,
    )
    matches = pattern.findall(src)
    assert not matches, (
        "Found answer call sites with a numeric truncation cap — the fix "
        f"has regressed: {matches}"
    )
