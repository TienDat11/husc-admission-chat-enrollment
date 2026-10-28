"""TDD: rag2025/src/domain/canonical/semantic_units.py — ticket 06.

Acceptance (``issues/06-deterministic-chunk-identity.md``):
- a unit's identity is structural (record type + field + slot) and does not
  depend on list position, iteration order or the clock;
- a major code (mã ngành), a subject combination (tổ hợp) and a timeline
  milestone (mốc thời gian) are atomic — no unit boundary may cut one;
- the atomicity check can actually *detect* a cut (negative control), otherwise
  a passing test would only prove the check is inert.

Everything is asserted through the module's public surface.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.record import (  # noqa: E402
    AuthorityLevel,
    CanonicalRecord,
    RecordStatus,
    RecordType,
)
from src.domain.canonical.semantic_units import (  # noqa: E402
    MA_NGANH,
    MOC_THOI_GIAN,
    TO_HOP,
    SemanticUnit,
    UnitKind,
    classify_atom,
    find_atomic_spans,
    semantic_units,
    split_atoms_across_units,
    split_field_units,
    split_text_units,
    structured_payload,
)

# The real text of a seeded TO_HOP record (ticket 28's store): one major code,
# five combinations, all inside a single sentence.
TO_HOP_TEXT = (
    "Tổ hợp xét tuyển năm 2026 ngành Quản lý văn hóa (mã ngành 7229042): "
    "C00, C03, C19, X70, D14 — C00: Ngữ văn, Lịch sử, Địa lý; "
    "C03: Ngữ văn, Toán, Lịch sử."
)
TO_HOP_VALUES = {
    "chi_tieu": 30,
    "ma_nganh": "7229042",
    "ten_nganh": "Quản lý văn hóa",
    "to_hop_codes": ["C00", "C03", "C19", "X70", "D14"],
}


def _record(**overrides) -> CanonicalRecord:
    fields = {
        "record_id": "TO_HOP:NGANH:7229042:2026",
        "record_type": RecordType.TO_HOP,
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-01-01",
        "status": RecordStatus.DRAFT,
        "authority_level": AuthorityLevel.HUSC_CANONICAL,
        "source_document_id": "THONG_TIN_TUYEN_SINH_202611.pdf",
        "source_page": 25,
        "text": TO_HOP_TEXT,
    }
    fields.update(overrides)
    return CanonicalRecord(**fields)


# ── atomic shape recognition ───────────────────────────────────────────────


def test_atomic_spans_find_code_combo_and_milestone():
    text = "Mã ngành 7229042, tổ hợp A00, hạn nộp 01/07/2026."
    found = {(span.kind, span.value) for span in find_atomic_spans(text)}
    assert (MA_NGANH, "7229042") in found
    assert (TO_HOP, "A00") in found
    assert (MOC_THOI_GIAN, "01/07/2026") in found


def test_atomic_spans_recognise_the_other_milestone_spellings():
    long_form = find_atomic_spans("Nộp trước ngày 01 tháng 07 năm 2026.")
    assert [s.value for s in long_form] == ["ngày 01 tháng 07 năm 2026"]
    iso = find_atomic_spans("Hiệu lực từ 2026-07-01.")
    assert [s.value for s in iso] == ["2026-07-01"]


def test_a_longer_digit_run_is_not_read_as_a_major_code():
    """Lookarounds keep a 12-digit id from masquerading as two major codes."""
    assert find_atomic_spans("Số 123456789012 không phải mã ngành.") == ()


def test_classify_atom_matches_whole_values_only():
    assert classify_atom("7229042") == MA_NGANH
    assert classify_atom("A00") == TO_HOP
    assert classify_atom("01/07/2026") == MOC_THOI_GIAN
    assert classify_atom("Quản lý văn hóa") is None
    assert classify_atom("mã 7229042 và 7229043") is None


# ── atomicity: no unit boundary may cut an atom ────────────────────────────


def test_no_atom_is_split_across_text_units():
    units = split_text_units(
        TO_HOP_TEXT, record_id="r", record_type="TO_HOP", budget_tokens=12
    )
    assert len(units) > 1, "the budget must actually force a split for this to test anything"
    assert split_atoms_across_units(TO_HOP_TEXT, units) == ()
    # Every combination survives verbatim in exactly one unit.
    for code in ("C00", "C03", "C19", "X70", "D14"):
        holders = [u for u in units if code in u.value]
        assert len(holders) >= 1
        assert all(code in u.value for u in holders)


def test_atomicity_check_detects_a_cut():
    """Negative control: the check must fail on a splitter that ignores atoms.

    A fixed-width cutter is the naive implementation this ticket exists to rule
    out. If ``split_atoms_across_units`` returned empty for it too, the positive
    test above would be vacuous.
    """
    text = "ab A00 yyy 01/07/2026"
    naive = _fixed_width_units(text, width=5)
    cut = split_atoms_across_units(text, naive)
    assert "A00" in cut
    assert "01/07/2026" in cut

    safe = split_text_units(text, record_id="r", record_type="TO_HOP", budget_tokens=3)
    assert split_atoms_across_units(text, safe) == ()


def test_paragraphs_become_separate_units_with_disjoint_spans():
    text = (
        "Ngành Quản lý văn hóa, mã ngành 7229042.\n\n"
        "Tổ hợp xét tuyển: A00, C03.\n\n"
        "Hạn nộp hồ sơ: 01/07/2026."
    )
    units = split_text_units(text, record_id="r", record_type="NGANH")
    assert [u.value for u in units] == [
        "Ngành Quản lý văn hóa, mã ngành 7229042.",
        "Tổ hợp xét tuyển: A00, C03.",
        "Hạn nộp hồ sơ: 01/07/2026.",
    ]
    spans = [u.source_span for u in units]
    assert spans == sorted(spans)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "spans must not overlap"
    assert split_atoms_across_units(text, units) == ()


def test_a_long_unbroken_paragraph_is_actually_split():
    """Regression: the word-budget path must cut, not emit one huge unit.

    A paragraph with no sentence terminator reaches the word-boundary splitter.
    An earlier version of that splitter never advanced its cursor, so it
    returned the whole paragraph as one unit — silently losing the size gate.
    """
    text = " ".join(["từ"] * 900)
    units = split_text_units(text, record_id="r", record_type="NGANH", budget_tokens=50)
    assert len(units) > 1
    assert "".join(u.value for u in units).count("từ") == 900
    assert max(len(u.value.split()) for u in units) <= 60


def test_a_multi_word_milestone_survives_the_word_boundary_splitter():
    """The case where the atom guard is load-bearing.

    ``ngày 01 tháng 07 năm 2026`` is one atom made of several words, so a
    word-budget splitter *can* land inside it. The negative control below is the
    obvious implementation — pack words to the budget, cut at word starts — and
    it does cut the milestone; the real splitter must not.
    """
    text = (
        ("Thí sinh cần lưu ý " * 4)
        + "hạn cuối là ngày 01 tháng 07 năm 2026 theo quy định "
        + ("của nhà trường " * 4)
        + "và mã ngành 7229042 áp dụng."
    )
    units = split_text_units(text, record_id="r", record_type="THOI_GIAN", budget_tokens=25)
    assert len(units) > 1, "the budget must force the word-boundary path"
    assert split_atoms_across_units(text, units) == ()
    assert sum("ngày 01 tháng 07 năm 2026" in u.value for u in units) == 1

    naive = _naive_word_budget_units(text, 25)
    assert split_atoms_across_units(text, naive) == ("ngày 01 tháng 07 năm 2026",), (
        "the negative control must actually cut the milestone, or this test "
        "proves nothing about the guard"
    )


def _naive_word_budget_units(text: str, budget: int) -> tuple[SemanticUnit, ...]:
    """A budget-only word splitter, used only as the atom-guard negative control."""
    from src.domain.canonical.semantic_units import _count_tokens, _encoder

    enc = _encoder()
    boundaries: list[int] = []
    current = 0
    for match in re.finditer(r"\S+", text):
        word_start, word_end = match.start(), match.end()
        if _count_tokens(text[current:word_end], enc) > budget and word_start > current:
            boundaries.append(word_start)
            current = word_start
    edges = [0, *boundaries, len(text)]
    return tuple(
        SemanticUnit(
            record_id="r",
            record_type="THOI_GIAN",
            field="text",
            slot=str(i),
            value=text[start:end],
            kind=UnitKind.TEXT,
            source_span=(start, end),
        )
        for i, (start, end) in enumerate(zip(edges, edges[1:]))
        if text[start:end].strip()
    )


def test_atomicity_audit_refuses_units_without_a_span():
    """The audit must not be able to pass by abstaining."""
    unspanned = (
        SemanticUnit(
            record_id="r",
            record_type="TO_HOP",
            field="text",
            slot="0",
            value="A00",
            kind=UnitKind.TEXT,
        ),
    )
    with pytest.raises(ValueError, match="source_span"):
        split_atoms_across_units("A00", unspanned)


def _fixed_width_units(text: str, width: int) -> tuple[SemanticUnit, ...]:
    """A deliberately naive splitter, used only as the atomicity negative control."""
    units = []
    for start in range(0, len(text), width):
        end = min(start + width, len(text))
        units.append(
            SemanticUnit(
                record_id="r",
                record_type="TO_HOP",
                field="text",
                slot=str(len(units)),
                value=text[start:end],
                kind=UnitKind.TEXT,
                source_span=(start, end),
            )
        )
    return tuple(units)


# ── structured units: one mã ngành / one tổ hợp per unit ───────────────────


def test_structured_payload_is_read_from_the_record():
    assert structured_payload(_record(values=TO_HOP_VALUES)) == TO_HOP_VALUES
    assert structured_payload(_record()) == {}


def test_each_subject_combination_is_its_own_atomic_unit():
    units = split_field_units(_record(values=TO_HOP_VALUES))
    combos = {u.slot: u for u in units if u.field == "to_hop_codes"}
    assert set(combos) == {"C00", "C03", "C19", "X70", "D14"}
    assert all(u.atomic for u in combos.values())
    assert combos["C19"].value == "tổ hợp xét tuyển: C19"


def test_major_code_is_its_own_atomic_unit():
    units = split_field_units(_record(values=TO_HOP_VALUES))
    codes = [u for u in units if u.field == "ma_nganh"]
    assert len(codes) == 1
    assert codes[0].atomic
    assert codes[0].value == "mã ngành: 7229042"


def test_a_milestone_field_is_its_own_atomic_unit():
    units = split_field_units(
        _record(record_type=RecordType.THOI_GIAN, values={"moc_thoi_gian": "01/07/2026"})
    )
    assert [u.value for u in units] == ["mốc thời gian: 01/07/2026"]
    assert units[0].atomic


def test_unknown_payload_keys_produce_no_unit():
    """A label the schema never declared must not be invented."""
    units = split_field_units(_record(values={"noi_dung_bi_an": "bí mật"}))
    assert units == ()


# ── structural, order-independent identity ─────────────────────────────────


def test_unit_key_is_record_type_field_and_slot_only():
    unit = SemanticUnit(
        record_id="r",
        record_type="TO_HOP",
        field="to_hop_codes",
        slot="A00",
        value="tổ hợp xét tuyển: A00",
        kind=UnitKind.ATOM,
        atomic=True,
    )
    assert unit.key == "TO_HOP|to_hop_codes|A00"


def test_combination_units_are_invariant_under_payload_permutation():
    """A set has no order: shuffling the payload must not renumber a unit."""
    forward = split_field_units(_record(values=TO_HOP_VALUES))
    reversed_values = {**TO_HOP_VALUES, "to_hop_codes": list(reversed(TO_HOP_VALUES["to_hop_codes"]))}
    backward = split_field_units(_record(values=reversed_values))
    assert [u.key for u in forward] == [u.key for u in backward]
    assert [u.value for u in forward] == [u.value for u in backward]


def test_units_are_invariant_under_payload_key_order():
    forward = split_field_units(_record(values=dict(TO_HOP_VALUES)))
    shuffled = {key: TO_HOP_VALUES[key] for key in reversed(list(TO_HOP_VALUES))}
    assert [u.key for u in split_field_units(_record(values=shuffled))] == [
        u.key for u in forward
    ]


def test_editing_one_combination_leaves_the_other_units_untouched():
    """The fine-grained scoping the chunk hash depends on, at unit level."""
    before = {u.key: u.value for u in split_field_units(_record(values=TO_HOP_VALUES))}
    edited_values = {**TO_HOP_VALUES, "to_hop_codes": ["C00", "C03", "C20", "X70", "D14"]}
    after = {u.key: u.value for u in split_field_units(_record(values=edited_values))}

    assert set(before) - set(after) == {"TO_HOP|to_hop_codes|C19"}
    assert set(after) - set(before) == {"TO_HOP|to_hop_codes|C20"}
    shared = set(before) & set(after)
    assert {k: before[k] for k in shared} == {k: after[k] for k in shared}


# ── whole-record unit set ──────────────────────────────────────────────────


def test_semantic_units_cover_fields_and_text():
    units = semantic_units(_record(values=TO_HOP_VALUES))
    fields = {u.field for u in units}
    assert "to_hop_codes" in fields and "ma_nganh" in fields and "text" in fields
    assert all(u.record_id == "TO_HOP:NGANH:7229042:2026" for u in units)
    assert all(u.record_type == "TO_HOP" for u in units)


def test_semantic_units_are_repeatable_within_a_process():
    record = _record(values=TO_HOP_VALUES)
    assert semantic_units(record) == semantic_units(record)


def test_a_record_without_a_payload_still_yields_its_text():
    units = semantic_units(_record())
    assert [u.field for u in units] == ["text"]
    assert units[0].value == TO_HOP_TEXT
