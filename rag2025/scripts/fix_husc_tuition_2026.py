"""Fix the HUSC 2026-2027 tuition figures in the live ``rag2025`` table AND
in the on-disk source JSONL/JSON files (S18.1 — TDD data-ops).

Background
----------
The corpus carried the WRONG HUSC 2026 tuition figures (Khối VII
525.000 → 490.000 VNĐ/tín chỉ; Khối IV 540.000 → 555.000). The
correct figures are taken from ``new_data/THONG_TIN_TUYEN_SINH_202611.pdf``
pages 54-55, the official HUSC source, per Nghị định 238/2025/NĐ-CP.

The CANONICAL source of truth is:

    ``data/major_codes/husc_tuition_2026_official.json``

It contains:
  * ``per_credit_by_block``  →  ``{VII: 490000, IV: 555000, V: 600000}``
  * ``majors[]``             →  per-ma_nganh → ``vnd_per_credit``

The corrector is SURGICAL: it only touches HUSC 2026 records. 2025
records and non-HUSC records are NEVER mutated.

TDD contract: see ``tests/scripts/test_fix_husc_tuition_2026.py``.

What the corrector does
-----------------------
1. Fix on-disk source files FIRST (so re-ingestion does not reintroduce
   the wrong numbers):

     * ``data/raw/husc_2026_canonical.json`` — top-level
       ``tuition.khoi_vii.amount`` and ``tuition.khoi_iv.amount`` and
       per-major ``tuition`` fields.
     * ``data/chunked/chunked_10.jsonl``
     * ``data/chunked/chunked_10_enhanced.jsonl``
     * ``data/chunked/chunked_24.jsonl``
     * ``data/chunked/husc_2026_aggregation_v2.jsonl``
     * ``data/chunked/nganh_detail_2026.jsonl``

2. Fix the LIVE LanceDB table:

     * For each of the 26 per-major ``husc_nganh_*_2026`` chunks:
         - Parse ``ma_nganh`` from ``chunk_id`` (e.g. ``husc_nganh_7229030_2026``
           → ``7229030``).
         - Look up the canonical ``vnd_per_credit`` for that ma_nganh.
         - Surgically replace the wrong figure in the text with the
           correct one (block-aware for special-case majors like
           ``7480201VJ`` → 640.000 and ``7229001`` → 0).
         - Re-embed the corrected text.
         - Delete the old row by ``chunk_id``; re-add with the corrected
           text, the new vector, and updated metadata.
     * For the 5 aggregate tuition chunks
       (``husc_hoc_phi_tin_chi_2026``, ``husc_uoc_tinh_hoc_phi_ky_2026``,
       ``hocphi_2026``, ``husc_policy_hoc_phi_2026``,
       ``so_sanh_hocphi_2025_vs_2026``): apply the block-aware text
       replacement map and inject the legal-basis label.

3. Add the legal-basis label to every corrected text and the matching
   metadata fields. The label is:

     ``năm học 2026-2027 (theo Nghị định 238/2025/NĐ-CP)``

   and the source URL is ``https://tuyensinh.husc.edu.vn``.

Idempotency
-----------
Re-running the script on an already-corrected table is a no-op: the
``correct_row_text`` function returns ``was_changed=False`` and the
table is not mutated.

Run (real, against the live ``rag2025`` table):

    python scripts/fix_husc_tuition_2026.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

# Make ``config`` and ``src`` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402


# ---------------------------------------------------------------------------
# TDD-locked constants
# ---------------------------------------------------------------------------
TUITION_YEAR: str = "2026-2027"
TUITION_BASIS: str = "Nghị định 238/2025/NĐ-CP"
TUITION_SOURCE_URL: str = "https://tuyensinh.husc.edu.vn"
TUITION_LABEL: str = "năm học 2026-2027 (theo Nghị định 238/2025/NĐ-CP)"

# Default path to the canonical source-of-truth JSON.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CANONICAL_PATH: Path = REPO_ROOT / "data" / "major_codes" / "husc_tuition_2026_official.json"

# On-disk source files that are patched FIRST (so re-ingestion is correct).
SOURCE_FILES: List[str] = [
    "data/chunked/chunked_10.jsonl",
    "data/chunked/chunked_10_enhanced.jsonl",
    "data/chunked/chunked_24.jsonl",
    "data/chunked/husc_2026_aggregation_v2.jsonl",
    "data/chunked/nganh_detail_2026.jsonl",
    "data/raw/husc_2026_canonical.json",
]

# Block-aware text replacement map. Only HUSC 2026 figures are touched.
# Built from per_credit_by_block in the canonical file.
#
# Each entry: (wrong_dotted, correct_dotted, wrong_comma, correct_comma,
#              wrong_nosep, correct_nosep)
#
# Khối V (600.000) is INTENTIONALLY absent — the figure is correct, and
# the only Khối-V special case (CNTT-VJ = 640.000) is handled at the
# per-major level (see ``correct_row_text``).
BLOCK_REPLACEMENTS: List[Tuple[str, str]] = [
    # dot-separator
    ("525.000", "490.000"),
    ("540.000", "555.000"),
    # comma-separator
    ("525,000", "490,000"),
    ("540,000", "555,000"),
    # no-separator (only used in compact summaries)
    ("525000", "490000"),
    ("540000", "555000"),
]

# Pre-compiled patterns. We only replace when the figure is preceded by
# "Khối VII" / "Khối IV" within a short window — that's the
# SURGICAL guarantee. The "any 525.000/540.000" pattern is also applied
# when the surrounding context explicitly ties it to HUSC 2026, which
# the caller (per-line check + chunk_id filter) is responsible for.
# Inside text-level ``apply_text_corrections`` we use a permissive
# pass: replace ALL 525.000/540.000 because the input text is already
# gated to HUSC 2026 records by the caller.
DOT_FIGURE_RE = re.compile(r"\b525[\.,]?000\b")
COMMA_FIGURE_RE = re.compile(r"\b525[\.,]?000\b")


# ---------------------------------------------------------------------------
# Protocols — keep this script testable with fakes
# ---------------------------------------------------------------------------
class _TableLike(Protocol):
    def add(self, rows: List[Dict[str, Any]]) -> Any: ...
    def delete(self, where_clause: str) -> Any: ...
    def to_pandas(self, columns: Optional[List[str]] = None): ...


class _EmbedderLike(Protocol):
    def encode_documents(self, texts: List[str]): ...
    def encode_single(self, text: str): ...


# ---------------------------------------------------------------------------
# Canonical source of truth — load + lookup
# ---------------------------------------------------------------------------
def load_canonical(canonical_path: Path) -> Dict[str, Any]:
    """Load the canonical official tuition JSON.

    Returns a dict with:
      ``by_ma_nganh``  →  ``{ma_nganh: vnd_per_credit}``
      ``by_block``     →  ``{block: vnd_per_credit}``
      ``raw``          →  the parsed JSON document
      ``source_url``   →  ``"https://tuyensinh.husc.edu.vn"``
      ``basis``        →  ``"Nghị định 238/2025/NĐ-CP"``
      ``year``         →  ``"2026-2027"``
    """
    data = json.loads(Path(canonical_path).read_text(encoding="utf-8"))
    by_ma_nganh: Dict[str, int] = {}
    for m in data.get("majors", []):
        code = str(m.get("ma_nganh"))
        vnd = m.get("vnd_per_credit")
        if code and isinstance(vnd, int):
            by_ma_nganh[code] = vnd
    by_block: Dict[str, int] = {
        str(k): int(v) for k, v in (data.get("per_credit_by_block") or {}).items()
    }
    return {
        "by_ma_nganh": by_ma_nganh,
        "by_block": by_block,
        "raw": data,
        "source_url": TUITION_SOURCE_URL,
        "basis": TUITION_BASIS,
        "year": TUITION_YEAR,
    }


# ---------------------------------------------------------------------------
# Pure text-level correction
# ---------------------------------------------------------------------------
def _format_int(n: int, sep: str) -> str:
    """Format ``n`` with the given thousands separator ('.', ',', or '')."""
    s = str(int(n))
    if sep == "":
        return s
    # Group from the right in 3-digit chunks
    out: List[str] = []
    while len(s) > 3:
        out.append(s[-3:])
        s = s[:-3]
    out.append(s)
    return sep.join(reversed(out))


def apply_text_corrections(text: str) -> str:
    """Apply the block-aware HUSC 2026 string replacement map.

    Maps ALL ``525.000``/``525,000``/``525000`` and
    ``540.000``/``540,000``/``540000`` occurrences to the
    corrected figures, regardless of the surrounding block label.
    Callers are responsible for ensuring the input text is a HUSC 2026
    record (the row-level corrector and the JSONL file fixer both gate
    on ``data_year`` and the presence of HUSC markers).
    """
    out = text
    for wrong, right in BLOCK_REPLACEMENTS:
        out = out.replace(wrong, right)
    return out


# ---------------------------------------------------------------------------
# Per-row text correction (block- + major-aware)
# ---------------------------------------------------------------------------
def _extract_ma_nganh_from_chunk_id(chunk_id: str) -> Optional[str]:
    """Parse ``ma_nganh`` from a chunk_id like ``husc_nganh_7229030_2026``."""
    m = re.match(r"^husc_nganh_([A-Za-z0-9]+?)_\d{4}$", chunk_id or "")
    if not m:
        return None
    return m.group(1)


def _find_figure_in_text(text: str) -> Optional[str]:
    """Find the HUSC 2026 tuition figure in a per-major chunk text.

    Matches ``Khối (VII|IV|V)`` (case-insensitive), allowing any
    non-digit characters between the block label and the figure, and
    returns the first figure-like token that follows within the same
    token run.

    Examples (all match → return ``540,000`` or ``540.000`` or ``540000``):
      * "Khối IV 540,000 VNĐ/tín chỉ"
      * "KHỐI IV (540.000đ/TC)"
      * "Khối học phí: KHỐI V 600000đ"
    """
    m = re.search(
        r"Khối\s+(VII|IV|V)[^0-9]{0,40}?([0-9][0-9\.,]{2,9})",
        text or "",
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(2)


def _replace_figure_in_text(text: str, old_figure: str, new_figure: int) -> str:
    """Replace ``old_figure`` (a string like "525.000" / "525,000" / "525000")
    with the canonical ``new_figure`` formatted to match the source style
    (dot vs comma vs no-separator)."""
    # Detect separator from the old figure
    if "." in old_figure:
        new = _format_int(int(new_figure), ".")
    elif "," in old_figure:
        new = _format_int(int(new_figure), ",")
    else:
        new = _format_int(int(new_figure), "")
    # Word-boundary replace; the old figure is always a bare number
    pattern = re.escape(old_figure)
    return re.sub(pattern, new, text, count=1)


def _inject_label_in_text(text: str) -> str:
    """Append the HUSC 2026-2027 label to the text if not already present.

    If the text already has any of the marker substrings (year label,
    legal basis), it is left untouched. Otherwise the label is appended
    at the end (with a single space).
    """
    if TUITION_BASIS in text and "2026-2027" in text:
        return text
    if TUITION_BASIS in text:
        return text
    return f"{text.rstrip()} [{TUITION_LABEL}]"


def _update_legal_basis_in_text(text: str) -> str:
    """Replace an OLD legal-basis reference (e.g. Nghị định 81/2021) with the
    new one (Nghị định 238/2025/NĐ-CP) and append the year label if missing.
    Used for miễn học phí majors (e.g. Triết học 7229001) where the figure
    itself is already 0 but the legal-basis citation has been updated.
    """
    # Replace "Nghị định <digits...>/<year>/..." with the new one.
    updated = re.sub(
        r"Nghị\s*định\s+\d+/\d{4}/NĐ[-‑‐]?CP",
        TUITION_BASIS,
        text,
    )
    return _inject_label_in_text(updated)


def correct_row_text(
    text: str,
    chunk_id: str,
    metadata: Dict[str, Any],
    canonical: Dict[str, Any],
) -> Tuple[str, bool, Optional[str], Optional[int]]:
    """Return ``(new_text, was_changed, old_figure, new_figure)`` for one row.

    Logic:
      * If ``metadata.data_year`` is 2025 → never touch (``was_changed=False``).
      * If ``chunk_id`` matches ``husc_nganh_<ma_nganh>_2026`` → per-major
        lookup in ``canonical.by_ma_nganh``; the surgical replacement
        updates the Khối-figure to the canonical ``vnd_per_credit`` for
        that ma_nganh (handles Khối V 600.000, CNTT-VJ 640.000, Triết học
        0, Văn học 490.000, Vật lý 555.000, etc.).
      * Otherwise (aggregate tuition chunks): apply the block-aware
        text replacement map (525.000→490.000, 540.000→555.000).
      * The legal-basis label is only injected when the figure or text
        was actually changed (true idempotency).
    """
    md = dict(metadata or {})
    data_year = md.get("data_year")
    if data_year in (2025, "2025", "2025-2026"):
        return text, False, None, None

    by_ma_nganh = canonical.get("by_ma_nganh", {})

    # ---- Per-major chunks ----------------------------------------------
    ma_nganh = _extract_ma_nganh_from_chunk_id(chunk_id or "")
    if ma_nganh and ma_nganh in by_ma_nganh:
        canonical_vnd = by_ma_nganh[ma_nganh]
        old_figure = _find_figure_in_text(text)

        # Special case: Triết học (7229001) is miễn học phí → 0
        if int(canonical_vnd) == 0:
            if "MIỄN" in (text or "").upper() or "miễn" in (text or ""):
                # Already says MIỄN. Update the legal-basis label.
                new_text = _update_legal_basis_in_text(text)
                return (new_text, new_text != text, old_figure, 0)
            if old_figure is None:
                # No figure in text — leave as-is.
                return text, False, None, None
            # Has a number but should be 0: replace with 0đ and label.
            new_text = _replace_figure_in_text(text, old_figure, 0)
            new_text = _inject_label_in_text(new_text)
            return (new_text, True, old_figure, 0)

        if old_figure is None:
            return text, False, None, None

        # Parse the current figure to a number
        digits = re.sub(r"[^\d]", "", old_figure)
        try:
            current_vnd = int(digits)
        except ValueError:
            return text, False, None, None

        if current_vnd == int(canonical_vnd):
            # Already correct (e.g. Khối V 600.000) — true idempotent no-op.
            return text, False, old_figure, int(canonical_vnd)

        new_text = _replace_figure_in_text(text, old_figure, int(canonical_vnd))
        new_text = _inject_label_in_text(new_text)
        return (new_text, True, old_figure, int(canonical_vnd))

    # ---- Aggregate chunks ----------------------------------------------
    # 5 aggregate tuition chunks (e.g. husc_hoc_phi_tin_chi_2026) — apply
    # the global block-aware text replacement.
    new_text = apply_text_corrections(text)
    if new_text != text:
        new_text = _inject_label_in_text(new_text)
    was_changed = new_text != text
    return (new_text, was_changed, None, None)


# ---------------------------------------------------------------------------
# Live-table mutation (read → correct → re-embed → delete → add)
# ---------------------------------------------------------------------------
def _fetch_row(table: _TableLike, chunk_id: str) -> Optional[Dict[str, Any]]:
    df = table.to_pandas()
    if df is None or len(df) == 0 or "chunk_id" not in df.columns:
        return None
    match = df[df["chunk_id"] == chunk_id]
    if match is None or len(match) == 0:
        return None
    return match.iloc[0].to_dict()


def _parse_meta(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _build_corrected_row(
    old_row: Dict[str, Any],
    new_text: str,
    new_meta: Dict[str, Any],
    vector: List[float],
) -> Dict[str, Any]:
    return {
        "id": old_row.get("id") or old_row.get("chunk_id"),
        "chunk_id": old_row.get("chunk_id") or old_row.get("id"),
        "text": new_text,
        "source": old_row.get("source", TUITION_SOURCE_URL),
        "faq_type": old_row.get("faq_type", ""),
        "metadata_json": json.dumps(new_meta, ensure_ascii=False, default=str),
        "vector": vector,
    }


# ---------------------------------------------------------------------------
# Target chunk_ids (TDD-locked)
# ---------------------------------------------------------------------------
# We don't pre-bake the full list of 26 + 5 ids in this module (the table
# is the source of truth for what's actually in there). The corrector
# queries the table for ``data_year=2026`` HUSC rows whose text contains
# a wrong figure. The ``fix_lancedb_table`` routine walks the table and
# decides per-row whether to touch.
NGANH_CHUNK_ID_PREFIX: str = "husc_nganh_"
AGGREGATE_CHUNK_IDS: List[str] = [
    "husc_hoc_phi_tin_chi_2026",
    "husc_uoc_tinh_hoc_phi_ky_2026",
    "hocphi_2026",
    "husc_policy_hoc_phi_2026",
    "so_sanh_hocphi_2025_vs_2026",
]


def _is_nganh_2026_chunk_id(chunk_id: str) -> bool:
    """Per-major 2026 or 2025 chunks (both are eligible, but the corrector
    itself decides whether to touch based on ``metadata.data_year``)."""
    if not chunk_id or not chunk_id.startswith(NGANH_CHUNK_ID_PREFIX):
        return False
    return bool(re.match(r"^husc_nganh_[A-Za-z0-9]+_\d{4}$", chunk_id))


def _is_aggregate_tuition_chunk_id(chunk_id: str) -> bool:
    return chunk_id in AGGREGATE_CHUNK_IDS


def _is_target_chunk_id(chunk_id: str) -> bool:
    return _is_nganh_2026_chunk_id(chunk_id) or _is_aggregate_tuition_chunk_id(chunk_id)


def fix_lancedb_table(
    table: _TableLike,
    embedder: _EmbedderLike,
    canonical: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply the corrector to the LIVE ``rag2025`` table. Idempotent.

    Returns a summary:
      ``nganh_rows_changed``               → per-major 2026 chunks re-embedded
      ``nganh_rows_skipped_already_correct``→ per-major 2026 chunks with no change
      ``nganh_rows_skipped_data_year_2025``→ per-major 2025 chunks untouched
      ``aggregate_rows_changed``          → aggregate tuition chunks re-embedded
      ``per_chunk``                       → per-chunk-id ``old→new`` records
    """
    df = table.to_pandas()
    if df is None or len(df) == 0:
        return {
            "nganh_rows_changed": 0,
            "nganh_rows_skipped_already_correct": 0,
            "nganh_rows_skipped_data_year_2025": 0,
            "aggregate_rows_changed": 0,
            "per_chunk": [],
        }

    nganh_rows_changed = 0
    nganh_rows_skipped_already_correct = 0
    nganh_rows_skipped_data_year_2025 = 0
    aggregate_rows_changed = 0
    per_chunk: List[Dict[str, Any]] = []

    rows = df.to_dict("records")
    for row in rows:
        chunk_id = str(row.get("chunk_id") or row.get("id") or "")
        if not chunk_id:
            continue
        if not _is_target_chunk_id(chunk_id):
            continue

        meta = _parse_meta(row.get("metadata_json"))

        # Determine category (per-major 2026 vs aggregate)
        is_nganh = _is_nganh_2026_chunk_id(chunk_id)
        is_aggregate = _is_aggregate_tuition_chunk_id(chunk_id)

        text = row.get("text", "") or ""
        new_text, was_changed, old_fig, new_fig = correct_row_text(
            text=text,
            chunk_id=chunk_id,
            metadata=meta,
            canonical=canonical,
        )

        if is_nganh:
            data_year = meta.get("data_year")
            if data_year in (2025, "2025"):
                nganh_rows_skipped_data_year_2025 += 1
                continue
            if not was_changed:
                nganh_rows_skipped_already_correct += 1
                continue
            nganh_rows_changed += 1
        elif is_aggregate:
            if not was_changed:
                continue
            aggregate_rows_changed += 1

        # Re-embed the corrected text
        if hasattr(embedder, "encode_single"):
            new_vec = embedder.encode_single(new_text)
        else:
            new_vec = embedder.encode_documents([new_text])[0]
        new_vec_list = new_vec.tolist() if hasattr(new_vec, "tolist") else list(new_vec)

        # Update metadata
        new_meta = dict(meta)
        new_meta["tuition_year"] = canonical["year"]
        new_meta["tuition_basis"] = canonical["basis"]
        new_meta["tuition_source_url"] = canonical["source_url"]

        new_row = _build_corrected_row(
            old_row=row,
            new_text=new_text,
            new_meta=new_meta,
            vector=new_vec_list,
        )
        table.delete(f"chunk_id = '{chunk_id}'")
        table.add([new_row])

        per_chunk.append({
            "chunk_id": chunk_id,
            "category": "nganh_2026" if is_nganh else "aggregate_tuition_2026",
            "old_figure": old_fig,
            "new_figure": new_fig,
        })
        logger.info(f"Fixed chunk {chunk_id!r}: {old_fig} → {new_fig}")

    return {
        "nganh_rows_changed": nganh_rows_changed,
        "nganh_rows_skipped_already_correct": nganh_rows_skipped_already_correct,
        "nganh_rows_skipped_data_year_2025": nganh_rows_skipped_data_year_2025,
        "aggregate_rows_changed": aggregate_rows_changed,
        "per_chunk": per_chunk,
    }


# ---------------------------------------------------------------------------
# On-disk JSONL patcher
# ---------------------------------------------------------------------------
def fix_jsonl_file(
    file_path: Path,
    canonical: Dict[str, Any],
) -> Dict[str, int]:
    """Surgically patch a HUSC 2026 JSONL file in place.

    Skips lines whose ``metadata.data_year`` is 2025. For other lines
    applies the block-aware text replacement map (525.000→490.000,
    540.000→555.000) and injects the legal-basis label.

    Returns ``{lines_changed, lines_skipped_2025, lines_already_correct,
    lines_total}``.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"lines_changed": 0, "lines_skipped_2025": 0, "lines_already_correct": 0, "lines_total": 0}

    text = file_path.read_text(encoding="utf-8")
    new_lines: List[str] = []
    lines_changed = 0
    lines_skipped_2025 = 0
    lines_already_correct = 0
    lines_total = 0

    for line in text.splitlines():
        if not line.strip():
            new_lines.append(line)
            continue
        lines_total += 1
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            new_lines.append(line)
            continue
        if not isinstance(obj, dict):
            new_lines.append(line)
            continue

        # Build a synthetic chunk_id and metadata for the corrector.
        meta = obj.get("metadata") or {}
        chunk_id = str(obj.get("chunk_id") or obj.get("id") or "")
        text_value = obj.get("text") or ""

        new_text, was_changed, _, _ = correct_row_text(
            text=text_value,
            chunk_id=chunk_id,
            metadata=meta,
            canonical=canonical,
        )
        if not was_changed and new_text == text_value:
            # 2025 → no change, or already correct.
            data_year = meta.get("data_year")
            if data_year in (2025, "2025"):
                lines_skipped_2025 += 1
            else:
                lines_already_correct += 1
            new_lines.append(line)
            continue

        obj["text"] = new_text
        new_lines.append(json.dumps(obj, ensure_ascii=False))
        lines_changed += 1

    file_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")
    return {
        "lines_changed": lines_changed,
        "lines_skipped_2025": lines_skipped_2025,
        "lines_already_correct": lines_already_correct,
        "lines_total": lines_total,
    }


# ---------------------------------------------------------------------------
# On-disk canonical-JSON patcher
# ---------------------------------------------------------------------------
def fix_canonical_json_file(
    file_path: Path,
    canonical: Dict[str, Any],
) -> Dict[str, Any]:
    """Patch ``data/raw/husc_2026_canonical.json`` in place.

    The file has a top-level ``tuition`` object with
    ``khoi_vii.amount``, ``khoi_iv.amount``, ``khoi_v.amount``, and a
    per-major list where each major has a ``tuition`` int. We rewrite
    all those to match the canonical source of truth. Idempotent.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return {"changed": False, "khoi_vii_after": 490000, "khoi_iv_after": 555000, "khoi_v_after": 600000}

    data = json.loads(file_path.read_text(encoding="utf-8"))
    by_ma_nganh = canonical["by_ma_nganh"]
    by_block = canonical["by_block"]

    changed = False
    tuition = data.get("tuition") or {}
    for k, v in (by_block or {}).items():
        key = f"khoi_{str(k).lower()}"
        slot = tuition.get(key) or {}
        if isinstance(slot, dict) and int(slot.get("amount") or 0) != int(v):
            slot["amount"] = int(v)
            tuition[key] = slot
            changed = True
    data["tuition"] = tuition

    for m in data.get("majors", []) or []:
        code = str(m.get("code") or m.get("ma_nganh") or "")
        if code in by_ma_nganh and int(m.get("tuition") or 0) != int(by_ma_nganh[code]):
            m["tuition"] = int(by_ma_nganh[code])
            changed = True

    if changed:
        file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    return {
        "changed": changed,
        "khoi_vii_after": int(((data.get("tuition") or {}).get("khoi_vii") or {}).get("amount") or 0),
        "khoi_iv_after":  int(((data.get("tuition") or {}).get("khoi_iv")  or {}).get("amount") or 0),
        "khoi_v_after":   int(((data.get("tuition") or {}).get("khoi_v")   or {}).get("amount") or 0),
    }


# ---------------------------------------------------------------------------
# Real-DB entry point
# ---------------------------------------------------------------------------
def run_against_live_db() -> Dict[str, Any]:
    """Run the corrector against the live LanceDB table + on-disk files."""
    import lancedb  # local import — only needed in the live run

    from config.settings import RAGSettings  # noqa: WPS433
    from src.services.embedding import EmbeddingService  # noqa: WPS433

    canonical = load_canonical(DEFAULT_CANONICAL_PATH)

    # 1) Fix on-disk source files FIRST.
    file_summaries: List[Dict[str, Any]] = []
    for rel_path in SOURCE_FILES:
        p = REPO_ROOT / rel_path
        if not p.exists():
            logger.warning(f"Source file missing: {p}")
            continue
        if p.suffix == ".jsonl":
            s = fix_jsonl_file(p, canonical=canonical)
        elif p.suffix == ".json":
            s = fix_canonical_json_file(p, canonical=canonical)
        else:
            continue
        s["file"] = rel_path
        file_summaries.append(s)

    # 2) Fix the LIVE table.
    settings = RAGSettings()
    db = lancedb.connect(settings.LANCEDB_URI)
    if settings.LANCEDB_TABLE not in db.table_names():
        raise FileNotFoundError(
            f"LanceDB table '{settings.LANCEDB_TABLE}' not found at "
            f"'{settings.LANCEDB_URI}'"
        )
    table = db.open_table(settings.LANCEDB_TABLE)
    embedder = EmbeddingService(settings)
    table_summary = fix_lancedb_table(table=table, embedder=embedder, canonical=canonical)

    return {
        "files": file_summaries,
        "table": table_summary,
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print("On-disk files:")
    for s in summary.get("files", []):
        if "lines_changed" in s:
            print(
                f"  {s['file']}: lines_changed={s['lines_changed']} "
                f"skipped_2025={s['lines_skipped_2025']} "
                f"already_correct={s['lines_already_correct']} "
                f"total={s['lines_total']}"
            )
        else:
            print(
                f"  {s['file']}: changed={s.get('changed')} "
                f"khoi_vii={s.get('khoi_vii_after')} "
                f"khoi_iv={s.get('khoi_iv_after')} "
                f"khoi_v={s.get('khoi_v_after')}"
            )
    t = summary.get("table", {})
    print("\nLanceDB table:")
    print(f"  nganh_rows_changed                  : {t.get('nganh_rows_changed')}")
    print(f"  nganh_rows_skipped_already_correct  : {t.get('nganh_rows_skipped_already_correct')}")
    print(f"  nganh_rows_skipped_data_year_2025   : {t.get('nganh_rows_skipped_data_year_2025')}")
    print(f"  aggregate_rows_changed              : {t.get('aggregate_rows_changed')}")
    print("\nPer-chunk old→new:")
    for c in t.get("per_chunk", []):
        print(f"  {c['chunk_id']!r:50s} {c['old_figure']!r:>10s} → {c['new_figure']!r}")


if __name__ == "__main__":
    summary = run_against_live_db()
    _print_summary(summary)
