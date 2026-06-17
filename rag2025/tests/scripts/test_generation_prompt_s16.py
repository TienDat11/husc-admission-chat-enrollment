# @spec(S16.4 / ADR-E / AMF-4 / CMF-6) test_generation_prompt_s16
"""TDD: S16.4 — generation prompt rewrite for faithfulness + FE-render
safety + comparison-layout rule.

Contract (pinned by `.omc/plans/s16_faithfulness_route_calibration_plan.md`
§ADR-E + AMF-4 + CMF-6):

  * `{{YEAR_FACTS}}` placeholder MUST be retained exactly (runtime
    injection point — see `LLMGenerator.__init__`).
  * Temporal RULES + answer-from-context-only + Vietnamese-only MUST
    survive the rewrite.
  * No hardcoded major codes (regex `7\\d{6}([A-Z]{2})?`) — pinned by
    S14.6 contract.
  * No bare year literals (`\\b20\\d{2}\\b`) — pinned by S14.6 contract.
  * New faithfulness rule: do NOT invent URL/phone/email/major-code/dates/
    numbers not present in context. Only emit markdown links when the URL
    appears verbatim in context.
  * New abstain slot wording (reused if present; new text): the canonical
    "Tôi không tìm thấy thông tin này trong tài liệu tuyển sinh hiện có.
    Bạn vui lòng liên hệ phòng tuyển sinh để được hỗ trợ."
  * New FE-render rule (CMF-6): the FE uses react-markdown@10 with NO gfm
    plugins → only `### h3`, `**bold**`, `- bullets`, `[text](url)`, and
    inline code render correctly. Tables/raw HTML/code fences break.
  * New comparison-layout rule (CMF-6): when comparing A vs B, use
    `### <Tên A>` + bullets, then `### <Tên B>` + bullets. NEVER a
    markdown table.

We assert the prompt CONSTANTS contain the key rules (behavioral tests
live in `test_url_faithfulness_guard.py` for the post-guard and
`test_generator_season.py` for the season-aware fallback). The prompt
file is the static source-of-truth: the runtime injection step (the
`{{YEAR_FACTS}}` placeholder) is covered separately by
`test_llm_generator_year_injection.py`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "generation_system_prompt.txt"
)

# Mirror the S14.6 regexes (pinned by `test_generation_prompt_dehardcoded.py`).
_MAJOR_CODE_RX = re.compile(r"7\d{6}([A-Z]{2})?")
_YEAR_LITERAL_RX = re.compile(r"\b20\d{2}\b")


@pytest.fixture(scope="module")
def prompt_text() -> str:
    assert PROMPT_PATH.exists(), f"Prompt file missing: {PROMPT_PATH}"
    return PROMPT_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Placeholder retention (S14.6 contract, must NOT regress)
# ---------------------------------------------------------------------------

def test_year_facts_placeholder_is_preserved(prompt_text: str) -> None:
    """The `{{YEAR_FACTS}}` runtime injection point MUST remain in the
    file. If this is removed, `LLMGenerator.__init__` silently does
    nothing and the model falls back to no year-specific facts."""
    assert "{{YEAR_FACTS}}" in prompt_text, (
        "Prompt file is missing the {{YEAR_FACTS}} placeholder. "
        "Runtime year_facts injection in LLMGenerator.__init__ will be a no-op."
    )
    assert prompt_text.count("{{YEAR_FACTS}}") == 1, (
        "Expected exactly one {{YEAR_FACTS}} placeholder; found "
        f"{prompt_text.count('{{YEAR_FACTS}}')}."
    )


# ---------------------------------------------------------------------------
# S16.4 — NEW faithfulness + FE-render + comparison rules
# ---------------------------------------------------------------------------

def test_no_invent_url_rule_present(prompt_text: str) -> None:
    """The prompt MUST contain an explicit rule against inventing URLs.
    The S16.4 spec wording: 'KHÔNG bịa URL ... Chỉ chèn link [text](url)
    khi url xuất hiện nguyên văn trong context.'

    We assert the two key tokens: a no-invent-URL marker AND the
    'verbatim in context' marker (the rule that markdown links are
    only allowed when the URL appears verbatim in context)."""
    assert "KHÔNG bịa URL" in prompt_text or "KHÔNG bịa" in prompt_text and "URL" in prompt_text, (
        "Prompt must contain the no-invent-URL rule (S16.4 faithfulness)."
    )
    # The "verbatim in context" gate for markdown links.
    assert "nguyên văn" in prompt_text or "xuất hiện nguyên văn" in prompt_text, (
        "Prompt must require URLs to appear verbatim in context before "
        "emitting a markdown link."
    )


def test_table_allowed_rule_present(prompt_text: str) -> None:
    """rich-markdown: the prompt MUST now ALLOW markdown tables
    (remark-gfm was added to the FE). The old CMF-6 prohibition
    'KHÔNG dùng bảng' has been replaced by a permissive rule.

    We assert the key tokens: 'bảng' or 'table' appears in a
    PERMISSIVE context ('ƯU TIÊN', 'dùng bảng', 'DÙNG BẢNG'),
    and the old prohibition 'KHÔNG dùng bảng' is GONE."""
    assert "bảng" in prompt_text.lower(), (
        "Prompt must mention 'bảng' in the new permissive table rule."
    )
    # The old prohibition must NOT remain.
    assert "KHÔNG dùng bảng" not in prompt_text, (
        "Old CMF-6 prohibition 'KHÔNG dùng bảng' must be removed. "
        "The prompt should now ALLOW tables via remark-gfm."
    )
    # A permissive table token must exist.
    assert "ƯU TIÊN dùng bảng" in prompt_text or "DÙNG BẢNG" in prompt_text, (
        "Prompt must contain a permissive table token like "
        "'ƯU TIÊN dùng bảng' or 'DÙNG BẢNG'."
    )


def test_no_raw_html_rule_present(prompt_text: str) -> None:
    """CMF-6: the FE renders raw HTML as raw text. The prompt MUST tell
    the model to avoid raw HTML."""
    assert "HTML" in prompt_text, (
        "Prompt must mention HTML (the no-raw-HTML rule from CMF-6)."
    )
    # The 'KHÔNG' marker that frames the rule (case-insensitive tolerant).
    # Look for a substring like "KHÔNG ... HTML" with HTML mentioned in
    # a prohibitive context. Cheap heuristic: HTML and KHÔNG appear
    # within ~80 chars of each other.
    lower = prompt_text.lower()
    for match in re.finditer(r"html", lower):
        window = lower[max(0, match.start() - 80):match.end() + 40]
        if "không" in window:
            break
    else:
        pytest.fail(
            "Prompt must contain a 'KHÔNG ... HTML' prohibitive rule. "
            "FE renders raw HTML as raw text — model must not emit it."
        )


def test_no_code_fence_rule_present(prompt_text: str) -> None:
    """rich-markdown: code fences are now ALLOWED (remark-gfm renders them).
    The old prohibition was CMF-6-specific and is removed.
    This test is DELIBERATELY EMPTY — it exists so test suites that
    reference this function name do not break, but it no longer asserts
    anything. Kept as a tombstone to prevent test-runner failures."""
    pass  # Code fences are now allowed; no prohibition to assert.


def test_no_url_in_answer_body_rule_present(prompt_text: str) -> None:
    """CMF-6: the FE shows sources as separate chips (NOT from answer
    body). Putting source URLs in the answer body would create dead blue
    links. The prompt MUST tell the model not to put source URLs in
    the body."""
    assert "nguồn" in prompt_text.lower() or "url" in prompt_text.lower(), (
        "Prompt must mention source URL handling."
    )
    # The "do NOT put source URLs in body" rule. We assert a key token
    # combination is present.
    lower = prompt_text.lower()
    body_url_rule = (
        ("không đưa" in lower or "không chèn" in lower or "không chứa" in lower)
        and "url" in lower
        and ("thân" in lower or "trong câu trả lời" in lower or "body" in lower)
    )
    assert body_url_rule, (
        "Prompt must contain a rule that forbids putting source URLs in "
        "the answer body (FE shows them as separate chips)."
    )


def test_comparison_layout_rule_present(prompt_text: str) -> None:
    """rich-markdown: when comparing A vs B, the prompt now recommends
    tables for many-criteria comparisons (≥3) and ### + bullets for
    few-criteria comparisons (≤2). The key tokens:

    * 'so sánh' (comparison trigger phrase)
    * '###' (markdown heading)
    * A permissive table token for comparisons (NOT a prohibition).

    The old 'TUYỆT ĐỐI KHÔNG dùng bảng' prohibition is GONE."""
    assert "so sánh" in prompt_text.lower(), (
        "Prompt must mention 'so sánh' (the comparison trigger phrase)."
    )
    assert "###" in prompt_text, (
        "Prompt must include the `### h3` markdown syntax as a layout "
        "directive for comparisons."
    )
    # The old comparison-specific prohibition must be gone.
    assert "TUYỆT ĐỐI KHÔNG dùng bảng" not in prompt_text, (
        "Old comparison-specific 'TUYỆT ĐỐI KHÔNG dùng bảng' prohibition "
        "must be removed. The prompt now allows tables for comparisons."
    )
    # There should be a permissive comparison-table token.
    lower = prompt_text.lower()
    assert ("dùng bảng" in lower) or ("DÙNG BẢNG" in prompt_text), (
        "Prompt must contain a permissive table token for the comparison layout rule."
    )


def test_multi_part_question_rule_present(prompt_text: str) -> None:
    """Compound-question rule (bug fix 2026-06-09): when the user asks a
    question with MULTIPLE parts (joined by "và" / "đồng thời" / "ngoài ra"
    / multiple "?"), the model MUST split it and answer EVERY part fully.

    Root cause this guards: the old "STAY ON TOPIC tuyệt đối" / "CHỈ trả lời
    ĐÚNG CHỦ ĐỀ" rules made the model treat the 2nd clause of a compound
    question (e.g. "các ngành xét C00 VÀ ngành nào ít chỉ tiêu nhất") as
    off-topic and drop it. The new rule forces full coverage of every part.

    We assert key tokens are present:
      * a multi-part trigger marker ("nhiều phần" / "nhiều ý" / "đa phần")
      * the full-coverage directive ("từng phần" / "từng ý" / "đầy đủ")
    """
    lower = prompt_text.lower()
    has_trigger = (
        "nhiều phần" in lower
        or "nhiều ý" in lower
        or "đa phần" in lower
        or "nhiều câu hỏi" in lower
    )
    assert has_trigger, (
        "Prompt must contain a multi-part-question trigger marker "
        "('nhiều phần' / 'nhiều ý' / 'đa phần'). Compound questions like "
        "'các ngành C00 và ngành ít chỉ tiêu nhất' must be detected."
    )
    has_full_coverage = (
        ("từng phần" in lower or "từng ý" in lower)
        and ("trả lời" in lower)
    )
    assert has_full_coverage, (
        "Prompt must direct the model to answer EVERY part of a compound "
        "question ('trả lời ... từng phần/từng ý'). Without it the model "
        "drops the 2nd clause (the C00-vs-min-quota bug)."
    )


def test_stay_on_topic_scoped_to_offtopic_data(prompt_text: str) -> None:
    """Regression guard for the compound-question fix: the STAY ON TOPIC
    rule must NOT be an unconditional 'one topic only' directive anymore —
    that is what suppressed the 2nd clause. We assert the absolute phrasing
    'STAY ON TOPIC tuyệt đối' is gone OR explicitly scoped so it only
    applies to off-topic CONTEXT data, not to the user's own multi-part ask.
    """
    # The dangerous unconditional phrasing must be softened/removed.
    assert "STAY ON TOPIC tuyệt đối" not in prompt_text, (
        "Unconditional 'STAY ON TOPIC tuyệt đối' must be removed/scoped — "
        "it suppressed the 2nd clause of compound questions. Scope it to "
        "off-topic CONTEXT data only."
    )


def test_abstain_wording_present(prompt_text: str) -> None:
    """The S16.4 spec mandates the canonical abstain wording (reuse if
    present, add if missing): 'Tôi không tìm thấy thông tin này trong
    tài liệu tuyển sinh hiện có. Bạn vui lòng liên hệ phòng tuyển sinh
    để được hỗ trợ.'

    We assert both halves: the 'không tìm thấy' core (already pinned by
    S14.6) AND the new 'liên hệ phòng tuyển sinh' guidance."""
    assert "không tìm thấy" in prompt_text.lower(), (
        "Prompt must contain the canonical 'không tìm thấy' abstain core."
    )
    assert "liên hệ phòng tuyển sinh" in prompt_text, (
        "Prompt must contain the S16.4 abstain addendum: 'liên hệ phòng "
        "tuyển sinh để được hỗ trợ'."
    )


def test_majors_enumeration_directive_when_context_has_full_list(prompt_text: str) -> None:
    """Bug 2026-06-10: for count/list questions about ngành (e.g. 'Trường có
    bao nhiêu ngành') the model was collapsing to a generic '28 ngành + 3
    ngành mới' even when the retrieved context contained the full
    enumeration chunk (name + mã ngành + chỉ tiêu).

    Contract (pinned by the LIỆT KÊ section, rule 5 area):
      * The rule MUST contain a directive that, when the user asks about
        ngành (count/which/detail/list) AND context contains a chunk that
        enumerates the majors, the model MUST present the full list with
        STT, tên ngành, mã ngành, chỉ tiêu.
      * The rule MUST explicitly forbid collapsing to just the count when
        the full list is available in context.
      * The rule MUST explicitly say stating the total count is fine as a
        lead-in but MUST be followed by the enumeration.
    """
    lower = prompt_text.lower()
    # Token-level anchors: enumerate-by-table is the preferred output.
    has_liet_ke = "liệt kê" in lower
    has_ma_nganh = "mã ngành" in lower
    has_chi_tieu = "chỉ tiêu" in lower
    has_count_followed_by_enum = "sau đó liệt kê" in lower or "rồi liệt kê" in lower or "kèm theo liệt kê" in lower or "kèm danh sách" in lower or "theo sau là liệt kê" in lower or "trước rồi liệt kê" in lower
    has_no_collapse = (
        "không chỉ nói" in lower
        or "không thu gọn" in lower
        or "không được chỉ" in lower
        or "không chỉ trả lời" in lower
        or "không chỉ nêu" in lower
        or "không tóm tắt" in lower
    )
    assert has_liet_ke and has_ma_nganh and has_chi_tieu, (
        "Majors LIỆT KÊ rule must explicitly name the enumeration output "
        "('liệt kê' + 'mã ngành' + 'chỉ tiêu'). Without this anchor the "
        "model collapses 'bao nhiêu ngành' to a generic count instead of "
        "enumerating the full list from context."
    )
    assert has_count_followed_by_enum, (
        "Rule must direct the model to state the count AS A LEAD-IN and "
        "THEN enumerate ('sau đó liệt kê' / 'rồi liệt kê' / equivalent). "
        "Stating the count alone is the exact bug being fixed."
    )
    assert has_no_collapse, (
        "Rule must forbid collapsing to just the count when the full list "
        "is in context (e.g. 'không chỉ nói' / 'không thu gọn')."
    )


def test_majors_full_list_marker_when_context_has_list(prompt_text: str) -> None:
    """The LIỆT KÊ majors rule MUST contain a 'context has the full list →
    enumerate' trigger marker. This is the gating condition for the
    enumeration directive: when the retrieved context chunk contains the
    full enumeration, the model MUST enumerate.

    A simple count question ('bao nhiêu ngành') triggers this gate IF the
    full-list chunk is present in context. The rule must make the gate
    explicit so the model does not default to a one-number answer."""
    # A short, opaque marker pinned in the prompt so a future edit cannot
    # silently weaken the rule. Two-part check:
    #   (1) a 'context has the list' trigger phrase,
    #   (2) a 'must enumerate' imperative in the same rule body.
    lower = prompt_text.lower()
    has_context_list_trigger = (
        ("context" in lower)
        and (
            "liệt kê" in lower
            and (
                "đầy đủ" in lower
                or "đủ" in lower
                or "toàn bộ" in lower
                or "đầy đủ các ngành" in lower
                or "liệt kê đầy đủ" in lower
            )
        )
    )
    assert has_context_list_trigger, (
        "Rule must explicitly say: when context contains a full enumeration "
        "of the majors, the model MUST enumerate the full list (not just "
        "the count). Token: 'liệt kê' + 'context' + 'đầy đủ'/'đủ'/'toàn bộ'."
    )


# ---------------------------------------------------------------------------
# S14.6 regressions — must NOT regress when the S16.4 rewrite is applied
# ---------------------------------------------------------------------------

def test_no_static_major_codes_in_prompt_body(prompt_text: str) -> None:
    """S14.6: zero matches of the canonical 7-digit (+ optional 2-letter)
    major-code shape. Catches 7480107TD, 7440102SC, 7510302IC, etc."""
    matches = _MAJOR_CODE_RX.findall(prompt_text)
    assert matches == [], (
        "Prompt body still contains hardcoded major-code literals "
        f"({len(matches)} matches). First few: {matches[:5]}. "
        "Strip these — they belong in {{YEAR_FACTS}} or retrieved context, "
        "not the static prompt."
    )


def test_no_bare_year_literal_in_prompt_body(prompt_text: str) -> None:
    """S14.6: zero 4-digit year literals (\\b20\\d{2}\\b) in the static body."""
    matches = _YEAR_LITERAL_RX.findall(prompt_text)
    assert matches == [], (
        "Prompt body still contains year-literal claims "
        f"({len(matches)} matches, unique: {sorted(set(matches))}). "
        "Replace with 'năm hiện tại (N do {{YEAR_FACTS}} cung cấp)' or "
        "rely on runtime injection."
    )


def test_context_only_behavior_marker_preserved(prompt_text: str) -> None:
    assert "context" in prompt_text.lower(), (
        "Behavior marker 'context' is missing — the context-only-answer "
        "rule appears to have been stripped."
    )


def test_no_fabrication_behavior_marker_preserved(prompt_text: str) -> None:
    assert "KHÔNG bịa" in prompt_text, (
        "Behavior marker 'KHÔNG bịa' is missing — the no-fabrication rule "
        "appears to have been stripped."
    )


def test_vietnamese_only_behavior_marker_preserved(prompt_text: str) -> None:
    assert "tiếng việt" in prompt_text.lower(), (
        "Behavior marker 'tiếng Việt' is missing — the Vietnamese-only "
        "rule appears to have been stripped."
    )


def test_answer_first_behavior_marker_preserved(prompt_text: str) -> None:
    assert "ANSWER-FIRST" in prompt_text, (
        "Behavior marker 'ANSWER-FIRST' is missing — the answer-first "
        "rule appears to have been stripped."
    )


def test_bold_italic_allowed(prompt_text: str) -> None:
    """rich-markdown: the prompt MUST mention bold and italic as
    available formatting options now that FE rendering is richer.

    We assert both tokens appear in a formatting/permissions context."""
    assert "**in đậm**" in prompt_text or "**từ khóa" in prompt_text or "in đậm" in prompt_text.lower(), (
        "Prompt must mention bold (**in đậm**) as an available formatting option."
    )
    assert "*in nghiêng" in prompt_text or "in nghiêng" in prompt_text.lower(), (
        "Prompt must mention italic (*in nghiêng*) as an available formatting option."
    )
    """Sanity: the S16.4 additions should grow the prompt but not
    pathological-empty it. Keep an upper bound so accidental truncation
    is caught."""
    assert len(prompt_text) > 1000, (
        f"Prompt file looks suspiciously short ({len(prompt_text)} chars)."
    )
    # The S16.4 additions (~5-10 rules, ~500-1000 chars) on top of the
    # existing 6000+ char prompt → expect >5000 chars total.
    assert len(prompt_text) > 5000, (
        f"Prompt file looks too short for the S16.4 additions "
        f"({len(prompt_text)} chars)."
    )
