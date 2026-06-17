"""Regression tests for gen_thi_sinh_pdf.py — guarantees the rendered PDF
report keeps every character of the source question/answer.

Background
----------
The 2 PDF reports (``86q_thi_sinh_report_full.pdf`` and
``86q_thi_sinh_report_thay_tuyen_sinh.pdf``) are the only artifact the
admissions office actually reads. The user's explicit requirement is that
NO character may be silently lost between the JSON results and the PDF.
Two prior sources of char-loss had to be guarded against:

1. **Question truncation caps**: the script used to call
   ``md_to_rl(r["question"], 200)`` and ``md_to_rl(r["question"], 100)``,
   cutting long Vietnamese questions mid-word. (Answers were already
   uncapped in a prior fix — locked in by ``test_gen_thi_sinh_pdf_full.py``.)
2. **Emoji-strip regex**: ``md_to_rl`` strips Unicode ranges to keep emoji
   from rendering as tofu boxes. Those ranges MUST NOT overlap Vietnamese
   code points or we lose đ, ô, ư, ạ, ề, ữ, ọ etc.

What we verify
--------------
- ``md_to_rl`` preserves a hard Vietnamese paragraph (đ, ô, ư, ạ, ề, ữ, ọ, ẵ)
- ``md_to_rl`` does NOT truncate a 500-char question to 200/100
- ``md_to_rl`` still strips globe/phone emoji while keeping the URL
- ``md_to_rl`` still converts **bold** → <b>bold</b>
- ``md_to_rl`` still escapes XML metachars (&, <, >)
- Source-code regression: no ``md_to_rl(r["question"], <int>)`` cap survives
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

pytest.importorskip("reportlab")

# --------------------------------------------------------------------------- #
# Locate the script. Tests live at rag2025/tests/scripts/, the script at
# rag2025/scripts/. Three ``parent`` hops land on the rag2025 package root.
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
# 1. The hardest Vietnamese must survive md_to_rl UNCHANGED.
# --------------------------------------------------------------------------- #
def test_md_to_rl_preserves_full_vietnamese(gen_module):
    """A dense Vietnamese paragraph containing every troublesome letter
    (đ, ô, ư, ạ, ề, ữ, ọ, ẵ, Ạ, Ễ, etc.) plus a decimal number with the
    currency mark (540.000đ) must pass through ``md_to_rl`` with every
    character intact.

    This guards against the emoji-strip regex being widened by mistake
    into ranges that overlap Vietnamese code points (U+00C0-U+024F and
    U+1EA0-U+1EF9) or combining diacritical marks (U+0300-U+036F).
    """
    text = (
        "Đại học Khoa học – Đại học Huế: ngành Công nghệ thông tin, "
        "tổ hợp A00/A01, điểm chuẩn 23,5; ưu tiên khu vực, đối tượng; "
        "học phí 540.000đ/tín chỉ. Quản trị, Kỹ thuật phần mềm, "
        "Vật lý, Hóa học, Sinh học, Địa lý."
    )

    result = gen_module.md_to_rl(text)

    # Every must-survive substring present in input must also be present
    # in output, byte-for-byte.
    must_contain = [
        "Đại",
        "ngành",
        "tổ hợp",
        "điểm chuẩn",
        "ưu tiên",
        "đối tượng",
        "540.000đ",
        "Kỹ thuật",
        "Vật lý",
        "Hóa học",
        "Địa lý",
        "Quản trị",
    ]
    for needle in must_contain:
        assert needle in result, (
            f"Vietnamese char/substring {needle!r} was dropped by "
            f"md_to_rl. Result: {result!r}"
        )

    # Stronger guarantee: the output's visible Vietnamese content
    # (every letter outside the dash separator) must be a perfect subset
    # of the input. Strip whitespace + the en-dash separator and compare.
    norm_in = re.sub(r"[\s\-–]", "", text)
    norm_out = re.sub(r"[\s\-–]", "", result)
    assert norm_in == norm_out, (
        f"md_to_rl changed the visible content. "
        f"Input normalized:  {norm_in!r}\n"
        f"Output normalized: {norm_out!r}"
    )


# --------------------------------------------------------------------------- #
# 2. A 500-char question must NOT be truncated to 200/100.
# --------------------------------------------------------------------------- #
def test_md_to_rl_no_limit_keeps_long_question(gen_module):
    """A 500-char question ends with a unique sentinel tail. The tail
    must survive ``md_to_rl(q)`` so the PDF can show the full question,
    not a 200- or 100-char prefix.
    """
    head = (
        "Cho em hỏi về điều kiện xét tuyển vào ngành Công nghệ thông tin "
        "của Trường Đại học Khoa học – Đại học Huế năm 2026, đặc biệt là "
        "tổ hợp A00/A01 và điểm chuẩn dự kiến, ưu tiên khu vực, đối "
    )
    tail = "iểm cộng, học phí, chỉ tiêu, lịch xét tuyển các đợt 1 2 3."
    filler = "X"  # padded only if head+tail < 500
    q = head + tail
    while len(q) < 500:
        q += filler
    q = q[:500]
    # Pin the tail to the very end of the string.
    q = q[: 500 - len(tail)] + tail
    assert len(q) == 500
    assert q.endswith(tail)

    result = gen_module.md_to_rl(q)

    # The tail substring must be in the result (we use its full form
    # with the original Vietnamese diacritics intact).
    assert tail in result, (
        f"md_to_rl truncated the question. Tail {tail!r} not found in "
        f"output (len {len(result)})."
    )
    # The 500-char input is plain text (no markdown, no emoji) so
    # md_to_rl should only collapse whitespace runs — output length
    # must be >= 500 minus a small whitespace-collapse margin.
    assert len(result) >= 480, (
        f"md_to_rl dropped too many chars: input=500, output={len(result)}."
    )


# --------------------------------------------------------------------------- #
# 3. Emoji are stripped, surrounding text + URL is preserved.
# --------------------------------------------------------------------------- #
def test_md_to_rl_strips_emoji_keeps_text(gen_module):
    """Globe/phone emoji should be removed; the Vietnamese label, the
    URL, and the word 'Hotline' must remain intact.
    """
    text = "🌐 Cổng tuyển sinh: https://tuyensinh.husc.edu.vn 📞 Hotline"

    result = gen_module.md_to_rl(text)

    # Vietnamese text + URL preserved.
    assert "Cổng tuyển sinh" in result, (
        f"Vietnamese label was dropped: {result!r}"
    )
    assert "Hotline" in result, f"'Hotline' was dropped: {result!r}"
    assert "https://tuyensinh.husc.edu.vn" in result, (
        f"URL was dropped: {result!r}"
    )
    # Emoji code points stripped.
    assert "🌐" not in result, f"globe emoji not stripped: {result!r}"
    assert "📞" not in result, f"phone emoji not stripped: {result!r}"


# --------------------------------------------------------------------------- #
# 4. **bold** still converts to <b>bold</b>.
# --------------------------------------------------------------------------- #
def test_md_to_rl_bold_conversion(gen_module):
    """``**Khối VII**`` must become ``<b>Khối VII</b>`` so ReportLab
    inline styling keeps working.
    """
    result = gen_module.md_to_rl("**Khối VII**")
    assert "<b>Khối VII</b>" in result, (
        f"Bold markdown conversion broken — got: {result!r}"
    )


# --------------------------------------------------------------------------- #
# 5. XML metachars (&, <, >) are still escaped before any tag injection.
# --------------------------------------------------------------------------- #
def test_md_to_rl_xml_escape(gen_module):
    """Raw ``&``, ``<``, ``>`` must be entity-escaped so ReportLab's
    Paragraph parser doesn't misinterpret them as markup. The function
    must not crash on these characters.
    """
    result = gen_module.md_to_rl("a < b & c > d")
    assert result == "a &lt; b &amp; c &gt; d", (
        f"XML escape broken — got: {result!r}"
    )
    # Stronger check: every raw metachar was escaped (and the count of
    # each escape sequence matches the count of source metachars).
    assert result.count("&lt;") == 1
    assert result.count("&amp;") == 1
    assert result.count("&gt;") == 1
    assert "<" not in result
    assert ">" not in result
    # Note: "&" alone won't be present because all & were turned into &amp;,
    # but the substring "&&" would mean we double-escaped — never the case.
    assert "&&" not in result


# --------------------------------------------------------------------------- #
# 6. Source-code regression: no positive int cap on QUESTION call sites.
# --------------------------------------------------------------------------- #
def test_no_question_truncation_call_sites_in_source():
    """Scan the script source. There must be NO call shaped like
    ``md_to_rl(r["question"], <positive int>)`` left. The fix is that
    every question rendering call drops its numeric 2nd argument; if
    someone re-adds one (200 / 100 / etc.), this test fails loudly.
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")

    # Match md_to_rl(...) where the 1st arg is r["question"] (or
    # r['question']) AND a numeric 2nd arg follows.
    pattern = re.compile(
        r"""md_to_rl\(
            \s*r\[(["'])question\1\]   # 1st arg: r["question"]
            \s*,\s*\d+                # 2nd arg: a bare integer
            \s*\)""",
        re.VERBOSE | re.DOTALL,
    )
    matches = pattern.findall(src)
    assert not matches, (
        "Found question call sites with a numeric truncation cap — the "
        f"fix has regressed: {matches}"
    )
