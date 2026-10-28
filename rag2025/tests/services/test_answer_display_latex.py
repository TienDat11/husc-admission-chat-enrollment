# @spec(latex-display-guard) test_answer_display_latex
r"""Regression guard for `_normalize_answer_display` (LaTeX in student answers).

Why this exists: the FE renders markdown via `react-markdown` + `remark-gfm`
with **no math plugin**, so any LaTeX the model emits is shown to the student
as literal markup. Measured in a real browser on 2026-09-13, a tuition question
returned this to the screen:

    $$\text{Học phí} = \text{Số tín chỉ đăng ký} \times 600.000\text{ VNĐ}$$

The generation prompt now bans raw LaTeX, but a prompt is a request, not a
guarantee — hence a deterministic post-guard. These tests pin the guard's
contract, including the deliberate decision to leave unknown commands visible
rather than half-translate them.
"""
from __future__ import annotations

import sys
from pathlib import Path

RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from services.llm_generator import _normalize_answer_display


# ---------------------------------------------------------------------------
# The measured production defect
# ---------------------------------------------------------------------------

def test_the_measured_tuition_formula_is_flattened() -> None:
    """The exact string that reached a student's screen, verbatim."""
    raw = (
        r"$$\text{Học phí} = \text{Số tín chỉ đăng ký} \times 600.000\text{ VNĐ}$$"
    )
    assert _normalize_answer_display(raw) == "Học phí = Số tín chỉ đăng ký × 600.000 VNĐ"


def test_no_latex_delimiters_survive() -> None:
    """No `$`, `$$`, `\\(`, `\\)`, `\\[`, `\\]` may remain in rendered output."""
    for raw in (
        r"$$x = 5$$",
        r"$x = 5$",
        r"\(x = 5\)",
        r"\[x = 5\]",
    ):
        out = _normalize_answer_display(raw)
        assert out == "x = 5", f"delimiters survived in {raw!r}: {out!r}"


def test_inline_math_inside_a_sentence_keeps_the_sentence() -> None:
    """Unwrapping must not eat surrounding prose."""
    assert _normalize_answer_display(r"giá \(4\) tỷ đồng") == "giá 4 tỷ đồng"


# ---------------------------------------------------------------------------
# Currency must survive — the failure mode of a naive `$...$` rule
# ---------------------------------------------------------------------------

def test_dollar_amounts_are_not_treated_as_math_delimiters() -> None:
    """Two currency marks must not pair into a phantom math span.

    A naive `\\$([^$]+)\\$` rule turns "giá $5 và $10" into "giá 5 và 10",
    silently deleting the currency marks from a tuition answer. The inline
    rule therefore requires the paired content to look like math.
    """
    assert _normalize_answer_display(r"giá $5 và $10") == r"giá $5 và $10"
    assert _normalize_answer_display(r"chi phí là $5, $10") == r"chi phí là $5, $10"


def test_a_single_dollar_amount_is_untouched() -> None:
    assert _normalize_answer_display(r"Học phí $5") == r"Học phí $5"


def test_inline_math_with_an_operator_is_still_unwrapped() -> None:
    """The currency guard must not disable genuine inline math."""
    assert _normalize_answer_display(r"$x = 5$") == "x = 5"
    assert _normalize_answer_display(r"$a \times b$") == "a × b"


# ---------------------------------------------------------------------------
# Command translation
# ---------------------------------------------------------------------------

def test_frac_becomes_a_slash() -> None:
    assert _normalize_answer_display(r"\frac{a}{b}") == "a/b"


def test_known_symbols_map_to_unicode() -> None:
    out = _normalize_answer_display(r"DT \leq 100 \geq 50 \neq 0")
    assert out == "DT ≤ 100 ≥ 50 ≠ 0"


def test_text_wrapper_is_unwrapped_not_dropped() -> None:
    """`\\text{VNĐ}` must keep its content — dropping it loses the unit."""
    assert _normalize_answer_display(r"\text{VNĐ} 2 tỷ") == "VNĐ 2 tỷ"


def test_nested_text_wrapper_is_unwrapped() -> None:
    assert _normalize_answer_display(r"\text{Số tín chỉ \times 2}") == "Số tín chỉ × 2"


# ---------------------------------------------------------------------------
# Deliberate non-translation: unknown input stays visible
# ---------------------------------------------------------------------------

def test_unknown_commands_are_left_verbatim() -> None:
    """A half-understood formula is worse than an obviously-broken one.

    Leaving `\\alpha` visible keeps the defect reportable instead of silently
    emitting a plausible-looking wrong symbol.
    """
    assert _normalize_answer_display(r"\alpha \beta") == r"\alpha \beta"


def test_unbalanced_frac_is_left_verbatim() -> None:
    assert _normalize_answer_display(r"\frac{a}{") == r"\frac{a}{"


# ---------------------------------------------------------------------------
# Non-interference — the guard must not touch ordinary answers
# ---------------------------------------------------------------------------

def test_plain_vietnamese_text_is_untouched() -> None:
    text = "Học phí ngành Kỹ thuật phần mềm là 600.000 VNĐ/tín chỉ. Thời gian đào tạo 4 năm."
    assert _normalize_answer_display(text) == text


def test_markdown_table_is_untouched() -> None:
    """Tables contain `|` and digits; the guard must not disturb them."""
    table = (
        "| Khối | Đơn giá (VNĐ/tín chỉ) |\n"
        "|------|----------------------|\n"
        "| V    | 600.000              |\n"
    )
    assert _normalize_answer_display(table) == table


def test_empty_and_none_like_input_is_safe() -> None:
    assert _normalize_answer_display("") == ""


def test_idempotent_on_already_clean_text() -> None:
    """Running the guard twice must equal running it once."""
    raw = r"$$\text{Học phí} = \text{Số tín chỉ} \times 600.000\text{ VNĐ}$$"
    once = _normalize_answer_display(raw)
    assert _normalize_answer_display(once) == once
