"""Claim-level citation contract (ticket 10).

Every high-impact claim in an answer (cutoff score / timeline / quota /
tuition) maps ``claim -> chunk -> canonical record -> source document`` plus
``page``/``section`` when available. The path deliberately reads only
chunk metadata (the lineage the canonical materializer writes), so it works
unchanged under a dense-only rollback -- hybrid fusion never touches it.

Contract rules encoded here:

* ``source_page`` and ``section`` are OPTIONAL on the model: a missing page
  must never break the answer surface (acceptance #3).
* Citations are PER CLAIM, never per table: a comparison table whose rows
  live in different chunks yields different ``chunk_id``/
  ``source_document_id`` per cell (acceptance #4).
* Ticket 33 constraint: a table piece carries the *labelled* row it contains,
  but ``rowspan_attribution[i]["row"]`` is a TABLE-scoped index while the
  piece's own text renumbers rows locally. Rows are tied to entries by the
  zip invariant (a piece holds exactly the entries of its rows, both in table
  order), and the anchor's label text is read from the entry's ``identity``
  -- it is never assumed to live in the same chunk. Unresolved rows expose
  their ``reason`` verbatim instead of letting the citation guess an
  identity.
"""
from __future__ import annotations

import re
from typing import Any, Literal, NamedTuple, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# The contract model (exported to the frontend via
# rag2025/scripts/export_citation_contract.py -- never hand-copied).
# ---------------------------------------------------------------------------

ClaimType = Literal["cutoff_score", "timeline", "quota", "tuition", "other"]


class ClaimCitation(BaseModel):
    """One high-impact claim traced to its source.

    ``grounded`` is False when no retrieved chunk contains the claim's value;
    abstention policy (ticket 11) consumes that flag -- it is the only field
    allowed to be False without invalidating the response.
    """

    claim: str = Field(..., description="Exact claim text as it appears in the answer")
    claim_type: ClaimType = Field(..., description="High-impact category of the claim")
    value: str = Field(..., description="Normalized value token (digits, decimalized)")
    chunk_id: Optional[str] = Field(default=None, description="Chunk that grounds the claim")
    record_id: Optional[str] = Field(default=None, description="Canonical record the chunk materialized from")
    record_version_key: Optional[str] = Field(default=None, description="Exact record version (release-scoped identity)")
    source_document_id: Optional[str] = Field(default=None, description="Source document the record was extracted from")
    source_page: Optional[int] = Field(default=None, description="Page in the source document; None when unknown -- never fabricated")
    section: Optional[str] = Field(default=None, description="Section/breadcrumb label inside the source document, when present")
    source_url: Optional[str] = Field(default=None, description="Canonical source URL when available")
    row_label: Optional[str] = Field(default=None, description="Identity label of the table row the claim sits in (resolved across split pieces)")
    note: Optional[str] = Field(default=None, description="Provenance caveat, e.g. the rowspan_unresolved reason")
    grounded: bool = Field(False, description="False means the claim value was not found in any retrieved chunk")


class Claim(NamedTuple):
    """A detected high-impact claim before it is traced to a chunk."""

    text: str  # verbatim fragment from the answer
    claim_type: str
    value: str  # normalized numeric value ("22.5", "4500000", "15/8/2025")


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

# Vietnamese decimal style: "22,5" (comma), thousands "4.500.000" (dot groups).
# The grouped alternative REQUIRES a dot and the plain one is word-anchored, so
# a 4-digit year is consumed whole ("2024" never splits into "202"+"4").
_NUM = re.compile(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?|\b\d+(?:[.,]\d+)?\b")

# keyword -> claim type; longest-first so "điểm xét tuyển" wins over "điểm".
_CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("điểm chuẩn", "cutoff_score"),
    ("điểm xét tuyển", "cutoff_score"),
    ("ngưỡng đầu vào", "cutoff_score"),
    ("học phí", "tuition"),
    ("chi phí học", "tuition"),
    ("chỉ tiêu", "quota"),
    ("mốc thời gian", "timeline"),
    ("nộp hồ sơ", "timeline"),
    ("xét tuyển", "timeline"),
]

# Any category keyword also terminates another keyword's scope: in
# "Học phí 20.000.000 đồng/năm, chỉ tiêu 1500" the tuition window must not
# swallow 1500 -- "chỉ tiêu" owns that number.
_KEYWORD_ALT = re.compile(
    "|".join(re.escape(k) for k, _ in _CATEGORY_KEYWORDS), re.IGNORECASE
)

# Standalone date tokens are timeline claims regardless of nearby keywords.
_DATE = re.compile(
    r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
    r"|\btháng\s+\d{1,2}(?:/\d{4})?\b"
    r"|\bnăm\s+\d{4}\b"
)
_WINDOW = 60  # chars after a keyword in which the value must appear
_MULTIPLIERS = {"triệu": 1_000_000.0, "tỷ": 1_000_000_000.0}


def normalize_number(raw: str, unit_window: str = "") -> str:
    """Normalize a Vietnamese-formatted number to a comparable decimal string.

    "22,5" -> "22.5"; "4.500.000" -> "4500000"; "16,5" + "triệu" -> "16500000".
    """
    s = raw
    # thousands-grouped with dots: 4.500.000
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
        s = s.replace(".", "")
    else:
        s = s.replace(",", ".")
    try:
        val = float(s)
    except ValueError:
        return s
    low = unit_window.lower()
    for word, mult in _MULTIPLIERS.items():
        if word in low:
            val *= mult
            break
    if val == int(val):
        return str(int(val))
    return str(val)

# A markdown table SEPARATOR row (dashes/colons only). Must contain a dash:
# a leaning data row like "| | A00 | 22,5 |" starts with "| |" and a looser
# pattern would silently drop it from _table_rows.
_SEP = re.compile(r"^\|(?:\s*:?-+:?\s*\|)+\s*$")

# Scope terminators for a category keyword: a claim list ends at a sentence
# end or a line break. A dot BETWEEN digits is a thousands separator
# ("4.500.000"), not a terminator, or every tuition claim would truncate to
# its first group. ``;`` is deliberately NOT a terminator -- the teacher
# comparison form ("Điểm chuẩn ngành CNTT 22,5; Kinh doanh 24,75") separates
# cells with semicolons, and cutting there would drop every cell but the first.
_BOUNDARY = re.compile(r"\.(?!\d)|[!?\n]")
# Claim-TEXT separator: unlike _BOUNDARY this DOES cut at ``;`` so each cell
# of a comparison sentence reads as its own fragment.
_CELL_SEP = re.compile(r"\.(?!\d)|[;!?\n]")
# A number right after one of these words is a year/date part, not a
# score/quota/tuition value ("Điểm chuẩn năm 2024 là 22,5" -> 2024 is timeline).
_CONTEXT_WORD = re.compile(r"(?:năm|tháng|ngày|đợt|mã|code)\s*$", re.IGNORECASE)


def _claim_value(ctype: str, raw: str, unit_window: str) -> str:
    """Canonical index key for a claim value.

    "15/8/2025" -> "15/8/2025"; "tháng 8/2025" -> "tháng8/2025";
    "năm 2025" -> "2025" (a bare year meets the chunk's bare year).
    """
    if ctype == "timeline" and _DATE.fullmatch(raw):
        token = re.sub(r"\s+", "", raw)
        return raw.split()[-1] if token.startswith("năm") else token
    return normalize_number(raw, unit_window)


def extract_claims(answer: str) -> list[Claim]:
    """Pull every high-impact claim (cutoff / timeline / quota / tuition).

    A category keyword scopes the run of text after it up to the next sentence
    or line break, and EVERY value in that run becomes a claim of the
    keyword's type. This is what makes the comparison form work:
    "Điểm chuẩn ngành CNTT 22,5; Kinh doanh 24,75" yields TWO cutoff_score
    claims -- one per cell -- so per-cell provenance (acceptance #4) is
    possible at all. A span already claimed by an earlier category is never
    re-claimed (score beats generic "xét tuyển"), and timeline keywords only
    ever produce date values -- never a bare number.
    Bare dates are timeline claims anywhere in the answer.

    Dedup is per (type, value, preceding context): the same number appearing
    in two different rows of one table stays two claims.
    """
    if not answer:
        return []
    found: list[tuple[int, Claim]] = []
    seen: set[tuple[str, str, str]] = set()
    claimed: list[tuple[int, int]] = []  # answer spans already turned into claims

    def _push(kw_start: int, pos: int, end: int, raw: str, ctype: str) -> None:
        if any(pos < e2 and p2 < end for p2, e2 in claimed):
            return
        value = _claim_value(ctype, raw, answer[end : end + 12])
        # Claim text: from the last separator (or keyword) up to the value, so
        # each cell of a comparison sentence reads as its own fragment.
        seg_start = max([kw_start] + [b.end() for b in _CELL_SEP.finditer(answer[kw_start:pos])])
        text = answer[seg_start:pos].strip() + " " + raw
        context = answer[max(kw_start, pos - 30) : pos].strip()
        key = (ctype, value, context)
        if key in seen:
            return
        seen.add(key)
        claimed.append((pos, end))
        found.append((pos, Claim(text=text.strip(), claim_type=ctype, value=value)))

    for keyword, ctype in _CATEGORY_KEYWORDS:
        for m in re.finditer(re.escape(keyword), answer, re.IGNORECASE):
            window = answer[m.end() : m.end() + _WINDOW]
            km = _KEYWORD_ALT.search(window)
            if km:
                window = window[: km.start()]
            bm = _BOUNDARY.search(window)
            scoped = window[: bm.start()] if bm else window
            if ctype == "timeline":
                for dm in _DATE.finditer(scoped):
                    _push(m.start(), m.end() + dm.start(), m.end() + dm.end(), dm.group(0), ctype)
                continue
            for nm in _NUM.finditer(scoped):
                raw = nm.group(0)
                before = scoped[nm.start() - 1] if nm.start() > 0 else ""
                if before == "/" or re.match(r"/\s*\d", scoped[nm.end() : nm.end() + 2]):
                    continue  # part of a date like 15/8/2025
                if before and re.fullmatch(r"[A-Za-z0-9]", before):
                    continue  # "A00" code: digits glued to a letter, not a value
                if _CONTEXT_WORD.search(scoped[: nm.start()]):
                    continue  # "năm 2024": the year is a timeline claim, not a score
                if re.fullmatch(r"(?:19|20)\d{2}", raw):
                    continue  # bare year: timeline, never a score/quota/tuition
                _push(m.start(), m.end() + nm.start(), m.end() + nm.end(), raw, ctype)

    for m in _DATE.finditer(answer):
        _push(m.start(), m.start(), m.end(), m.group(0), "timeline")

    found.sort(key=lambda p: p[0])
    return [c for _, c in found]


# ---------------------------------------------------------------------------
# Chunk-side value index
# ---------------------------------------------------------------------------


def _chunk_text(chunk: dict) -> str:
    return str(chunk.get("text") or chunk.get("summary") or "")


def _chunk_values(chunk: dict) -> set[str]:
    """Normalized values present in a chunk (body numbers, dates, identities).

    Timeline claims carry date tokens ("15/8/2025", "tháng8"), so the chunk
    side indexes them too -- otherwise every date claim would read as
    ungrounded. Identity labels inherited via attribution may carry numbers.
    """
    text = _chunk_text(chunk)
    out: set[str] = set()
    for m in _NUM.finditer(text):
        out.add(normalize_number(m.group(0), text[m.end() : m.end() + 12]))
    for m in _DATE.finditer(text):
        token = re.sub(r"\s+", "", m.group(0))
        out.add(m.group(0).split()[-1] if token.startswith("năm") else token)
    for entry in (chunk.get("metadata") or {}).get("rowspan_attribution") or []:
        ident = str(entry.get("identity") or "")
        for m in _NUM.finditer(ident):
            out.add(normalize_number(m.group(0)))
    return out


def _table_rows(text: str) -> list[tuple[int, str]]:
    """(local_data_row_index, row text) for markdown table lines of one piece.

    The chunker re-emits ``header + separator`` at the top of every piece
    (``to_chunk_text``), so the header is identified structurally -- the pipe
    line immediately followed by the separator -- never by a content
    heuristic. Indices here are PIECE-local; tying them to the table-scoped
    ``rowspan_attribution[i]["row"]`` is ``_row_context``'s job.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    pipes = [i for i, ln in enumerate(lines) if ln.startswith("|")]
    start = 0
    for i in pipes:
        if i + 1 in pipes and _SEP.match(lines[i + 1]):
            start = i + 2
            break
    rows: list[tuple[int, str]] = []
    idx = 0
    for i in pipes:
        if i < start or _SEP.match(lines[i]):
            continue
        rows.append((idx, lines[i]))
        idx += 1
    return rows


# ---------------------------------------------------------------------------
# Provenance resolution
# ---------------------------------------------------------------------------


def source_provenance(chunk: dict) -> dict:
    """Lineage keys for one chunk, tolerating legacy (non-canonical) rows.

    Reads the ``lineage`` block the canonical chunk builder writes (ticket 06)
    first and falls back to flat ``metadata`` keys, so a chunk from either
    producer resolves the same way. Returns record_id / record_version_key /
    source_document_id / source_page / section / source_url -- each None when
    absent. A missing page is a missing page, never a broken response
    (acceptance #3).
    """
    md = chunk.get("metadata") or {}
    lin = chunk.get("lineage") or {}

    def _pick(key: str) -> Any:
        return lin.get(key) if lin.get(key) is not None else md.get(key)

    section = _pick("section") or md.get("section")
    if not section:
        crumbs = md.get("breadcrumbs") or chunk.get("breadcrumbs")
        if isinstance(crumbs, list) and crumbs:
            section = str(crumbs[-1])
    page = _pick("source_page")
    if page is not None:
        try:
            page = int(page)
        except (TypeError, ValueError):
            page = None
    return {
        "record_id": _pick("record_id"),
        "record_version_key": _pick("record_version_key"),
        "source_document_id": _pick("source_document_id"),
        "source_page": page,
        "section": str(section) if section else None,
        "source_url": _pick("source_url"),
    }


def _row_cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _is_leaning(row: str) -> bool:
    """A data row whose leading (identity) cell is empty but which carries a value.

    Mirrors the chunker's candidate rule (``first_content > 0``): the first
    non-empty cell is not column 0, so the row leans on a merged cell above it.
    An all-empty row is ``first_content is None`` in the chunker and yields no
    provenance entry, so it is excluded here too -- otherwise the row-to-entry
    zip below would misalign.
    """
    cells = _row_cells(row)
    return bool(cells) and not cells[0] and any(cells)


def _row_context(chunk: dict, value: str) -> tuple[Optional[str], Optional[str]]:
    """(row_label, note) for the table row in ``chunk`` that holds ``value``.

    A piece's local row numbers are NOT the table-scoped indices its provenance
    entries use (ticket 33: each piece carries only its own rows but labels them
    with the whole table's indices). The mapping is recovered by zipping the
    piece's *leaning* rows, in order, with its entries sorted by table-scoped
    row: ticket 33 guarantees a piece holds exactly the entries of the rows it
    contains, so both lists are the same length and in the same order. When that
    invariant cannot be checked, the label comes from the row's own first cell if
    present and is never guessed.

    ``row_label`` is the attribution entry's ``identity`` -- the anchor's label
    text the chunker already resolved and stored on THIS piece, so the anchor is
    never assumed to live in this chunk (ticket 33). ``note`` carries an
    unresolved row's ``reason`` verbatim; the citation reads it, it does not
    infer.
    """
    md = chunk.get("metadata") or {}
    attributed = [e for e in (md.get("rowspan_attribution") or []) if isinstance(e, dict)]
    unresolved = [e for e in (md.get("rowspan_unresolved") or []) if isinstance(e, dict)]
    entries = sorted(
        [("_a", e) for e in attributed] + [("_u", e) for e in unresolved],
        key=lambda pair: int(pair[1].get("row", -1)),
    )
    rows = _table_rows(_chunk_text(chunk))
    leaning = [i for i, text in rows if _is_leaning(text)]
    zipped = len(leaning) == len(entries)
    for i, text in rows:
        norm = {normalize_number(m.group(0), text[m.end() : m.end() + 12]) for m in _NUM.finditer(text)}
        if value not in norm:
            continue
        cells = _row_cells(text)
        if cells and cells[0]:
            return cells[0], None
        if zipped and i in leaning:
            kind, entry = entries[leaning.index(i)]
            if kind == "_a":
                return (str(entry.get("identity") or "") or None), None
            return None, (str(entry.get("reason") or "") or None)
        return None, None
    return None, None


# ---------------------------------------------------------------------------
# Citation builder
# ---------------------------------------------------------------------------

_CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "cutoff_score": ("điểm chuẩn", "xét tuyển", "điểm"),
    "tuition": ("học phí", "đồng", "thu"),
    "quota": ("chỉ tiêu", "tuyển"),
    "timeline": ("tháng", "năm", "ngày", "đợt", "lịch"),
}


def _score_chunk(chunk: dict, claim: Claim) -> int:
    """Prefer chunks that carry BOTH the value and the category vocabulary."""
    text = _chunk_text(chunk).lower()
    score = 0
    if any(h in text for h in _CATEGORY_HINTS.get(claim.claim_type, ())):
        score += 1
    return score


def build_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Trace every claim in ``answer`` to chunk -> record -> source.

    Returns a list of ``ClaimCitation`` dicts (one per claim, in answer
    order). Citations are per-claim by construction: two claims whose values
    live in different chunks resolve to different ``chunk_id`` /
    ``source_document_id`` -- a comparison table can therefore never share
    one source id for the whole table (acceptance #4).
    """
    claims = extract_claims(answer)
    if not claims:
        return []
    value_index: list[tuple[str, dict]] = []
    for c in chunks:
        for v in _chunk_values(c):
            value_index.append((v, c))

    out: list[dict] = []
    for claim in claims:
        matches = [c for v, c in value_index if v == claim.value]
        if matches:
            best = max(matches, key=lambda c: _score_chunk(c, claim))
            label, note = _row_context(best, claim.value)
            prov = source_provenance(best)
            citation = ClaimCitation(
                claim=claim.text,
                claim_type=claim.claim_type,  # type: ignore[arg-type]
                value=claim.value,
                chunk_id=str(best.get("chunk_id") or best.get("id") or "") or None,
                record_id=prov["record_id"],
                record_version_key=prov["record_version_key"],
                source_document_id=prov["source_document_id"],
                source_page=prov["source_page"],
                section=prov["section"],
                source_url=prov["source_url"],
                row_label=label,
                note=note,
                grounded=True,
            )
        else:
            # Ungrounded: abstention (ticket 11) triggers on this; the page
            # field stays None because nothing was found, not because a
            # source lacks a page.
            citation = ClaimCitation(
                claim=claim.text,
                claim_type=claim.claim_type,  # type: ignore[arg-type]
                value=claim.value,
                grounded=False,
            )
        out.append(citation.model_dump())
    return out
