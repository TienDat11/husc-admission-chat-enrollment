"""Semantic units — the atomic, structurally-addressed pieces of a record.

Ticket 06, see ``.scratch/husc-rag-admin-upgrade/issues/06-deterministic-chunk-identity.md``.
This module answers one question: *what is a unit of meaning inside one canonical
record version?* It does not hash anything (``src/services/chunk_identity.py``
does) and it does not build chunks (``src/services/canonical_chunk_builder.py``).

Why structure, not content
--------------------------
A unit's identity is ``record_type | field | slot``. Nothing in that key is a
list position, an iteration order, a wall-clock value or a hash of the corpus:

* ``record_type`` is the record's declared fact kind;
* ``field`` is the name of the field the unit was read from (``text``,
  ``ma_nganh``, ``to_hop_codes``, …) — a name the schema owns;
* ``slot`` is ``"0"`` for a scalar field, and **the item's own canonical value**
  for a set-like (list) field, never the item's index.

That last point is the one that matters. ``to_hop_codes`` is a *set* of subject
combinations: its order in the payload is the author's typing order, not
information. Indexing it by position would make every unit after an edit shift,
so changing ``C19`` to ``C20`` would renumber ``D14`` and ``X70`` and hand them
new chunk ids — a change the content never made. Keying the slot by the value
makes the unit set invariant under permutation of the input list *and* keeps an
edit to one combination confined to that combination's unit.

The atomic-unit rule
--------------------
Three shapes are **one unit and MUST NOT be split**, at any level:

* a major code (mã ngành) — ``7229042``;
* a subject combination (tổ hợp) — ``A00``, ``X70``, ``D14``;
* a timeline milestone (mốc thời gian) — ``01/07/2026``,
  ``ngày 01 tháng 07 năm 2026``, ``2026-07-01``.

:func:`find_atomic_spans` locates them in free text; :func:`split_text_units`
refuses every boundary that would land strictly inside one; and
:func:`split_atoms_across_units` is the auditor — it returns the atoms that some
unit boundary cut in half, so a test can assert the list is empty and a
deliberately naive splitter can prove the assertion actually detects a cut.

Layering
--------
Domain-only: no I/O, no settings, no chunker import. ``tiktoken`` is used for the
paragraph budget when importable and a whitespace proxy is used when it is not —
both are deterministic, so a unit set never depends on which one was installed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from src.domain.canonical.record import CanonicalRecord

#: Bumped when the *rules below* change (a new atomic shape, a different slot
#: scheme). It is not a content hash and not a timestamp; the chunking code's
#: ``CHUNKER_VERSION`` (``src/services/chunk_identity.py``) is the stamp that
#: reaches lineage and manifests, and it must be bumped whenever this one is.
UNIT_RULES_VERSION = "1.0"

#: Separator inside a unit key. The three components are enum values and field
#: names, so the separator never occurs in one — the key stays unambiguous.
SEP = "|"

#: Paragraph budget, in tokens. A paragraph at or below this stays one unit;
#: a longer one is split at sentence boundaries first. Matches the chunker's
#: default target (``config.settings.CHUNK_SIZE_TOKENS``) without importing it —
#: the domain layer must not depend on application settings.
DEFAULT_UNIT_TOKEN_BUDGET = 350

# ── atomic shapes ──────────────────────────────────────────────────────────

# A major code: 6-7 digits standing alone. The lookarounds refuse a slice of a
# longer run, so a 12-digit tax id is never mistaken for two major codes.
_MA_NGANH_RE = re.compile(r"(?<![0-9A-Za-z])\d{6,7}(?![0-9A-Za-z])")
# A subject combination: one uppercase letter + two digits ("A00", "C03",
# "X70", "D14"). Vietnamese combination codes are exactly this shape.
_TO_HOP_RE = re.compile(r"(?<![0-9A-Za-z])[A-Z]\d{2}(?![0-9A-Za-z])")
# A timeline milestone, in the three spellings the corpus actually carries.
_DATE_NUMERIC_RE = re.compile(r"(?<![\d/])\d{1,2}[/-]\d{1,2}[/-]\d{4}(?![\d/])")
_DATE_LONG_RE = re.compile(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}")
_DATE_ISO_RE = re.compile(r"(?<![\d-])\d{4}-\d{2}-\d{2}(?![\d-])")

MA_NGANH = "ma_nganh"
TO_HOP = "to_hop"
MOC_THOI_GIAN = "moc_thoi_gian"

#: Longest-first is not needed: overlaps are resolved by the sort in
#: :func:`find_atomic_spans`, which keeps the earliest start and, among equal
#: starts, the longest match.
_ATOM_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (MA_NGANH, _MA_NGANH_RE),
    (TO_HOP, _TO_HOP_RE),
    (MOC_THOI_GIAN, _DATE_LONG_RE),
    (MOC_THOI_GIAN, _DATE_NUMERIC_RE),
    (MOC_THOI_GIAN, _DATE_ISO_RE),
)

#: Shape tests used to classify a *structured* field value (no lookarounds — a
#: payload value must be the atom and nothing else).
_ATOM_FULLMATCH: tuple[tuple[str, re.Pattern[str]], ...] = (
    (MA_NGANH, re.compile(r"\d{6,7}")),
    (TO_HOP, re.compile(r"[A-Z]\d{2}")),
    (MOC_THOI_GIAN, re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}")),
    (MOC_THOI_GIAN, re.compile(r"\d{4}-\d{2}-\d{2}")),
    (MOC_THOI_GIAN, re.compile(r"ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}")),
)

#: Vietnamese labels for the structured fields a unit may be read from. A key
#: absent from this map produces no structured unit — inventing a label for an
#: unknown key would assert a meaning the schema never declared. The record's
#: ``text`` unit still carries whatever the payload said.
FIELD_LABELS: Mapping[str, str] = {
    "ma_nganh": "mã ngành",
    "ten_nganh": "tên ngành",
    "chi_tieu": "chỉ tiêu",
    "diem_trung_tuyen": "điểm trúng tuyển",
    "hoc_phi_tin_chi_vnd": "học phí mỗi tín chỉ (VNĐ)",
    "thoi_gian_dao_tao_nam": "thời gian đào tạo (năm)",
    "phuong_thuc": "phương thức xét tuyển",
    "khoi_nganh": "khối ngành",
    "to_hop_codes": "tổ hợp xét tuyển",
    "to_hop_code": "tổ hợp xét tuyển",
    "moc_thoi_gian": "mốc thời gian",
    "deadline": "hạn cuối",
    "ngay": "mốc thời gian",
    "thoi_gian": "mốc thời gian",
}

#: Fields whose value is a *set*: one unit per item, and the item's own value is
#: the slot. Order in the payload is not information.
SET_VALUED_FIELDS = frozenset({"to_hop_codes", "to_hop_code", "moc_thoi_gian", "ngay"})

#: Keys inside a record's structured payload that carry the field values. The
#: canonical schema is additive (``extra="allow"``), so a structured payload
#: lives beside ``text`` rather than in a declared field.
_STRUCTURED_PAYLOAD_KEYS = ("values", "fields", "facts")


class UnitKind(str, Enum):
    """Where a unit came from — the shape of the thing it asserts."""

    #: A structured assertion read from the record's payload ("mã ngành: 7229042").
    ATOM = "atom"
    #: A span of the record's own ``text``.
    TEXT = "text"
    #: A table chunk produced by the table chunker (ticket 32/33 reuse).
    TABLE = "table"


@dataclass(frozen=True)
class AtomicSpan:
    """One occurrence of an atomic shape inside free text."""

    kind: str  # MA_NGANH | TO_HOP | MOC_THOI_GIAN
    value: str
    start: int
    end: int


@dataclass(frozen=True)
class SemanticUnit:
    """One atomic, structurally-addressed piece of a record version.

    ``source_span`` is the half-open ``[start, end)`` range of the record's
    ``text`` this unit was cut from, or ``None`` when the unit did not come from
    ``text`` (a structured atom, or a table chunk whose text is the chunker's).
    It exists so atomicity is *checkable* rather than asserted: a boundary that
    lands inside a major code is visible in the spans, not merely absent from
    the rendered values.
    """

    record_id: str
    record_type: str  # RecordType value (a str, so a key never depends on the enum)
    field: str
    slot: str
    value: str
    kind: UnitKind
    atomic: bool = False
    source_span: tuple[int, int] | None = None

    @property
    def key(self) -> str:
        """Structural identity: ``record_type | field | slot``. No content, no index."""
        return f"{self.record_type}{SEP}{self.field}{SEP}{self.slot}"


# ── atomic spans ───────────────────────────────────────────────────────────


def find_atomic_spans(text: str) -> tuple[AtomicSpan, ...]:
    """Every atomic occurrence in ``text``, earliest first, never overlapping.

    Overlaps cannot arise from the current patterns (a date and a combination
    code share no shape), but the resolver is explicit so adding a pattern later
    cannot silently produce two spans over one range.
    """
    candidates: list[AtomicSpan] = []
    for kind, pattern in _ATOM_PATTERNS:
        for match in pattern.finditer(text):
            candidates.append(AtomicSpan(kind, match.group(0), match.start(), match.end()))

    candidates.sort(key=lambda s: (s.start, -(s.end - s.start)))
    resolved: list[AtomicSpan] = []
    cursor = -1
    for span in candidates:
        if span.start < cursor:
            continue
        resolved.append(span)
        cursor = span.end
    return tuple(resolved)


def classify_atom(value: str) -> str | None:
    """The atomic kind of a *whole* structured value, or None if it is not one.

    Whole-value matching, not search: ``"7229042"`` is a major code, while
    ``"mã 7229042 và 7229043"`` is a sentence that happens to contain codes —
    that one belongs to the text splitter, which keeps both codes whole.
    """
    candidate = value.strip()
    if not candidate:
        return None
    for kind, pattern in _ATOM_FULLMATCH:
        if pattern.fullmatch(candidate):
            return kind
    return None


def split_atoms_across_units(
    text: str, units: Iterable[SemanticUnit]
) -> tuple[str, ...]:
    """The atoms in ``text`` that no single unit contains whole.

    The auditor behind the atomicity acceptance test. Empty means every major
    code, combination and milestone in ``text`` sits inside exactly one unit.
    A unit without a ``source_span`` cannot be audited, so it is an error rather
    than a silent pass — the check must not be able to succeed by abstaining.
    """
    materialised = tuple(units)
    for unit in materialised:
        if unit.source_span is None:
            raise ValueError(
                f"unit {unit.key!r} has no source_span — atomicity cannot be "
                "audited on a unit that does not say where it came from"
            )
    cut: list[str] = []
    for span in find_atomic_spans(text):
        whole = any(
            start <= span.start and span.end <= end
            for start, end in (u.source_span for u in materialised)  # type: ignore[misc]
        )
        if not whole:
            cut.append(span.value)
    return tuple(cut)


# ── text splitting ─────────────────────────────────────────────────────────

_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")
# Colon is deliberately NOT a terminator: on these records it introduces the
# assertion ("... (mã ngành 7229042): C00, C03") and cutting there would orphan
# the value from the field it belongs to.
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.;!?])\s+")


def _count_tokens(text: str, enc) -> int:
    if enc is None:
        # Whitespace proxy. Deterministic, and only ever compared against the
        # same budget the caller passed, so it cannot make a run non-reproducible.
        return len(text.split())
    return len(enc.encode(text))


def _encoder():
    """cl100k_base when tiktoken is installed, else None (whitespace proxy)."""
    try:  # pragma: no cover - exercised implicitly by whichever env is present
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - tiktoken is installed in this repo
        return None


def _snap_boundaries(boundaries: Sequence[int], spans: Sequence[AtomicSpan], end: int) -> list[int]:
    """Move every boundary that lands strictly inside an atomic span to its end.

    This is the rule the ticket states as "mã ngành, tổ hợp, mốc thời gian are
    atomic units and are not cut across". A boundary at ``span.start`` or
    ``span.end`` is legal — it is a boundary *between* units; one strictly
    inside splits the atom.
    """
    snapped: list[int] = []
    for boundary in boundaries:
        moved = boundary
        for span in spans:
            if span.start < moved < span.end:
                moved = span.end
        if 0 < moved < end and (not snapped or moved > snapped[-1]):
            snapped.append(moved)
    return snapped


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    """``[start, end)`` of every non-blank paragraph, in text order."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_BREAK_RE.finditer(text):
        spans.append((cursor, match.start()))
        cursor = match.end()
    spans.append((cursor, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _split_oversized(
    text: str, start: int, end: int, spans: Sequence[AtomicSpan], budget: int, enc
) -> list[tuple[int, int]]:
    """Cut a span that is over budget at word boundaries, never inside an atom.

    Reached only when a single sentence is longer than the whole budget. The
    word scan keeps the split points off whitespace runs, and
    :func:`_snap_boundaries` moves any that still lands inside an atom.
    """
    boundaries: list[int] = []
    current = start
    for match in re.finditer(r"\S+", text[start:end]):
        word_start = start + match.start()
        word_end = start + match.end()
        # Cut *before* the word that would overflow, never inside a word — and
        # only when at least one word sits after the previous cut, so a single
        # word longer than the whole budget is emitted rather than dropped.
        if _count_tokens(text[current:word_end], enc) > budget and word_start > current:
            boundaries.append(word_start)
            current = word_start
    boundaries = _snap_boundaries(boundaries, spans, end)

    pieces: list[tuple[int, int]] = []
    previous = start
    for boundary in boundaries:
        pieces.append((previous, boundary))
        previous = boundary
    pieces.append((previous, end))
    return [(s, e) for s, e in pieces if text[s:e].strip()]


def split_text_units(
    text: str,
    *,
    record_id: str,
    record_type: str,
    field: str = "text",
    budget_tokens: int = DEFAULT_UNIT_TOKEN_BUDGET,
    enc=None,
) -> tuple[SemanticUnit, ...]:
    """Cut ``text`` into units that never split an atomic shape.

    Paragraphs first (paragraph order is meaning, so the slot is the paragraph
    index); a paragraph over ``budget_tokens`` is cut at sentence boundaries; a
    sentence still over budget is cut at word boundaries. Every cut point is
    passed through :func:`_snap_boundaries`, so no major code, combination or
    milestone can straddle two units.
    """
    if enc is None:
        enc = _encoder()
    spans = find_atomic_spans(text)

    units: list[SemanticUnit] = []
    for para_start, para_end in _paragraph_spans(text):
        candidates = [para_start, para_end]
        if _count_tokens(text[para_start:para_end], enc) > budget_tokens:
            candidates = [para_start]
            candidates.extend(
                para_start + m.end()
                for m in _SENTENCE_BREAK_RE.finditer(text[para_start:para_end])
            )
            candidates.append(para_end)

        boundaries = [para_start]
        boundaries.extend(
            b for b in _snap_boundaries(candidates, spans, para_end) if b > para_start
        )
        if boundaries[-1] != para_end:
            boundaries.append(para_end)

        for start, end in zip(boundaries, boundaries[1:]):
            if _count_tokens(text[start:end], enc) > budget_tokens:
                pieces = _split_oversized(text, start, end, spans, budget_tokens, enc)
            else:
                pieces = [(start, end)]
            for piece_start, piece_end in pieces:
                value = text[piece_start:piece_end].strip()
                if not value:
                    continue
                units.append(
                    SemanticUnit(
                        record_id=record_id,
                        record_type=record_type,
                        field=field,
                        # Slot is the unit's ordinal inside this field. For text
                        # that ordinal *is* structure: paragraph 2 of a record is
                        # a different assertion from paragraph 1.
                        slot=str(len(units)),
                        value=value,
                        kind=UnitKind.TEXT,
                        source_span=(piece_start, piece_end),
                    )
                )
    return tuple(units)


# ── structured fields ──────────────────────────────────────────────────────


def structured_payload(record: CanonicalRecord) -> Mapping[str, object]:
    """The record's structured field values, or an empty map.

    The canonical schema is additive and ``extra="allow"``, so a record written
    by the extractor carries ``values`` (a dict) beside ``text``. A record with
    no such payload — every hand-authored fact — simply yields no structured
    units; the text units still describe it.
    """
    extra = record.model_extra or {}
    for key in _STRUCTURED_PAYLOAD_KEYS:
        candidate = extra.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _render(label: str, item: object) -> str:
    if isinstance(item, bool):
        return f"{label}: {'có' if item else 'không'}"
    if isinstance(item, float) and item.is_integer():
        return f"{label}: {int(item)}"
    return f"{label}: {item}"


def split_field_units(
    record: CanonicalRecord, payload: Mapping[str, object] | None = None
) -> tuple[SemanticUnit, ...]:
    """One unit per structured assertion the record's payload declares.

    Keys are visited in sorted order so a differently-ordered payload yields the
    same unit sequence, and a set-valued field contributes one unit per item
    with the item's own value as its slot (see the module docstring).
    """
    if payload is None:
        payload = structured_payload(record)
    record_type = record.record_type.value
    units: list[SemanticUnit] = []

    for key in sorted(payload):
        label = FIELD_LABELS.get(key)
        if label is None:
            continue
        raw = payload[key]
        items: list[object]
        if isinstance(raw, (list, tuple)):
            # Sorted by canonical string form: the unit set is invariant under
            # any permutation of the payload's list.
            items = sorted(raw, key=lambda item: str(item))
            slots = [str(item) for item in items]
        elif isinstance(raw, Mapping) or raw is None or raw == "":
            continue
        else:
            items = [raw]
            slots = ["0"]

        for item, slot in zip(items, slots):
            rendered = _render(label, item)
            atom_kind = classify_atom(str(item))
            units.append(
                SemanticUnit(
                    record_id=record.record_id,
                    record_type=record_type,
                    field=key,
                    slot=slot,
                    value=rendered,
                    kind=UnitKind.ATOM if atom_kind else UnitKind.TEXT,
                    atomic=atom_kind is not None,
                )
            )
    return tuple(units)


def semantic_units(
    record: CanonicalRecord,
    *,
    budget_tokens: int = DEFAULT_UNIT_TOKEN_BUDGET,
    enc=None,
    include_fields: bool = True,
) -> tuple[SemanticUnit, ...]:
    """Every semantic unit of one record version, structured units first.

    The record's ``text`` is always split into units. The structured payload
    adds one unit per declared field — which is what makes a major code, a
    combination or a milestone *its own* unit rather than merely an unbroken
    substring of a sentence.
    """
    units: list[SemanticUnit] = []
    if include_fields:
        units.extend(split_field_units(record))
    units.extend(
        split_text_units(
            record.text,
            record_id=record.record_id,
            record_type=record.record_type.value,
            budget_tokens=budget_tokens,
            enc=enc,
        )
    )
    return tuple(units)
