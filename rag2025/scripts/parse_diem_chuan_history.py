# @spec(PHASE-A1) deterministic parser for điểm chuẩn history HTML files
"""Parse ``data/raw_thongbao/diem_chuan_*.html`` into a normalized JSON history.

Input files (3):
  - ``diem_chuan_2024_thi.html`` — điểm chuẩn 2024 (xét thi THPT)
  - ``diem_chuan_2024_hb.html``   — điểm chuẩn 2024 (xét học bạ)
  - ``diem_chuan_2025_full.html`` — điểm chuẩn 2025 (4 phương thức: THPT, HB, ĐGNL HN, ĐGNL TPHCM)

Schema per entry (one row per major × year × method):
  {
    "major_code": str,                # e.g. "7480201", "7480201VJ"
    "major_name": str,                # friendly Vietnamese name (from husc_majors_current.json + canonical)
    "year": int,                      # 2024, 2025
    "method": "thi_thpt" | "hoc_ba",  # current implementation focuses on these two; others noted in ``_meta``
    "to_hop": list[str] | None,       # tổ hợp; only known for 2026 (we surface from canonical when available)
    "diem_chuan": float,              # the thang-điểm-30 score
    "_source_file": str,              # which HTML file the row came from (debug)
    "_row_index": int,                # row position in the source table (debug)
  }

Mapping strategy:
  - Use ``services.major_code_validator.get_whitelist()`` as the canonical
    allowlist (year-loadable, prefers 2026.json).
  - Resolve friendly names from ``husc_majors_current.json`` +
    ``raw/husc_2026_canonical.json`` (union).
  - Rows whose major code is NOT in the whitelist are SKIPPED but LOGGED
    (the user explicitly forbade silent drops).
  - 2024 HTML files contain only an ``<h3>`` title + an image of the table
    (no parseable tabular data in the HTML body). The parser logs this
    fact and emits ZERO rows for those files; the recommender downstream
    operates on whatever year(s) the parser could actually extract.

Pure-deterministic, no LLM, no network, no I/O outside the input HTML +
output JSON.
"""
from __future__ import annotations

import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Allow ``from services....`` when invoked as a script from rag2025/ root.
_HERE = Path(__file__).resolve().parent
_RAG_ROOT = _HERE.parent
_SRC = _RAG_ROOT / "src"
if str(_RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_RAG_ROOT))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from services.major_code_validator import get_whitelist  # noqa: E402

# ---- Paths ------------------------------------------------------------------
_RAW_DIR = _RAG_ROOT / "data" / "raw_thongbao"
_OUT_PATH = _RAG_ROOT / "data" / "major_codes" / "diem_chuan_history.json"

# Files in priority: 2025 (full table) FIRST so the deduplication picks the
# most recent + richest source when the same code appears in 2024.
_SOURCES: List[Tuple[str, int, str]] = [
    # (filename, year, method)
    ("diem_chuan_2025_full.html", 2025, "thi_thpt"),  # also emits hoc_ba rows from same table
    ("diem_chuan_2024_thi.html", 2024, "thi_thpt"),
    ("diem_chuan_2024_hb.html", 2024, "hoc_ba"),
]

# ---- Name resolution --------------------------------------------------------
_NAME_BY_CODE: Dict[str, str] = {}
_TO_HOP_BY_CODE: Dict[str, List[str]] = {}


def _load_name_maps() -> None:
    """Load friendly name + tổ hợp maps from the canonical data files.

    Order:
      1. ``husc_majors_current.json`` (alias-aware, crawler-updatable)
      2. ``raw/husc_2026_canonical.json`` (PDF-grounded; 26 programs)

    Both files are pure-JSON, no network.
    """
    global _NAME_BY_CODE, _TO_HOP_BY_CODE
    _NAME_BY_CODE = {}
    _TO_HOP_BY_CODE = {}

    # Source 1
    p1 = _RAG_ROOT / "data" / "major_codes" / "husc_majors_current.json"
    if p1.exists():
        try:
            d = json.loads(p1.read_text(encoding="utf-8"))
            for m in d.get("majors", []):
                if isinstance(m, dict) and m.get("ma_nganh") and m.get("ten"):
                    _NAME_BY_CODE.setdefault(m["ma_nganh"], m["ten"])
        except Exception:
            pass

    # Source 2
    p2 = _RAG_ROOT / "data" / "raw" / "husc_2026_canonical.json"
    if p2.exists():
        try:
            d = json.loads(p2.read_text(encoding="utf-8"))
            for prog in d.get("programs", []):
                if isinstance(prog, dict) and prog.get("code"):
                    code = prog["code"]
                    if prog.get("name"):
                        _NAME_BY_CODE.setdefault(code, prog["name"])
                    if prog.get("to_hop"):
                        _TO_HOP_BY_CODE[code] = list(prog["to_hop"])
        except Exception:
            pass


def _code_to_name(code: str) -> str:
    return _NAME_BY_CODE.get(code, code)  # fallback: code itself (logged downstream)


# ---- HTML table extractor (stdlib only) -------------------------------------

class _TableParser(HTMLParser):
    """Collect rows of a single <table> → list[list[str]] of cell text.

    The 2025 file has exactly one <tbody><tr>...</tr>...</tbody> structure;
    we ignore <thead>. We DO NOT depend on a 7-column shape — we accept any
    row length and let the caller pick indices defensively.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[str]] = []
        self._cur_row: Optional[List[str]] = None
        self._cur_cell: Optional[List[str]] = None
        self._depth: int = 0  # how many <tr>/<td> we are inside

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: D401
        if tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = []
        elif tag in ("p", "br", "div", "strong", "em", "b", "i", "span"):
            # Insert a separator so multi-paragraph cells stay parseable.
            if self._cur_cell is not None:
                self._cur_cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._cur_row is not None:
            self.rows.append(self._cur_row)
            self._cur_row = None
        elif tag in ("td", "th"):
            if self._cur_cell is not None and self._cur_row is not None:
                txt = "".join(self._cur_cell).strip()
                txt = re.sub(r"\s+", " ", txt)
                self._cur_row.append(txt)
            self._cur_cell = None

    def handle_data(self, data: str) -> None:
        if self._cur_cell is not None:
            self._cur_cell.append(data)


def _extract_table_rows(html: str) -> List[List[str]]:
    """Return ALL table rows from every <table> in the document.

    We only keep rows that have at least one digit (a candidate điểm cell).
    """
    parser = _TableParser()
    parser.feed(html)
    return [r for r in parser.rows if any(re.search(r"\d", c) for c in r)]


# ---- Cell-text → score helpers ---------------------------------------------

_NUM_RX = re.compile(r"-?\d+(?:[.,]\d+)?")


def _to_float(s: str) -> Optional[float]:
    """Convert a Vietnamese-formatted score to float, or None on failure.

    Accepts '16.00', '16,50', '1.500,00' (rejected — we only want 0..30 scale).
    """
    if not s:
        return None
    s = s.replace("\xa0", " ").strip()
    m = _NUM_RX.search(s)
    if not m:
        return None
    raw = m.group(0).replace(",", ".")
    try:
        v = float(raw)
    except ValueError:
        return None
    # Sanity: thang-điểm-30 is in [0, 30] for THPT/HB; ĐGNL scores are 0..1500.
    # We keep any positive number, the caller decides if it's in-scale.
    return v if v >= 0 else None


def _looks_like_major_code(s: str) -> bool:
    """True for strings like '7480201' or '7480201VJ'."""
    return bool(re.fullmatch(r"7\d{6}(?:[A-Z]{2})?", s.strip()))


# ---- Main parse logic per file ---------------------------------------------

def _parse_2025_full(html: str) -> Tuple[List[dict], List[dict]]:
    """Parse the 2025 full table.

    Columns (verified by reading diem_chuan_2025_full.html):
      0: Số TT
      1: Tên trường, Ngành học (may have extra <p> or <em>)
      2: Mã ngành
      3: THPT (thang 30)
      4: Học bạ (thang 30)
      5: ĐGNL ĐHQG TPHCM (thang 1200)
      6: ĐGNL ĐHQG Hà Nội (thang 150)

    We extract two entries per row: (thi_thpt, diem=THPT), (hoc_ba, diem=HB).
    The ĐGNL columns are NOT in the schema (we focus on the thang-30 scores).
    """
    rows = _extract_table_rows(html)
    emitted: List[dict] = []
    skipped: List[dict] = []
    # 2025 file: first 2 rows are the duplicated thead (we should skip them)
    # because they have the column labels in <thead>. Our parser only keeps
    # rows with digits, but the thead has 30/1200/150 in the headers, so
    # it WILL show up. Drop the first row (labels).
    started = False
    row_index = 0
    for r in rows:
        if len(r) < 7:
            continue
        # The header row contains 'Số TT' or 'Tên trường' — skip it
        joined = " ".join(r).lower()
        if not started:
            if "số tt" in joined or "tên trường" in joined or "mã ngành" in joined:
                continue
            started = True
        row_index += 1
        major_name_raw = r[1]
        major_code_raw = r[2]
        if not _looks_like_major_code(major_code_raw):
            skipped.append({
                "row_index": row_index,
                "reason": "no_major_code",
                "name": major_name_raw,
                "raw_row": r,
            })
            continue
        code = major_code_raw.strip()
        thi = _to_float(r[3])
        hb = _to_float(r[4])
        if thi is not None:
            emitted.append({
                "major_code": code,
                "major_name": _code_to_name(code),
                "year": 2025,
                "method": "thi_thpt",
                "to_hop": list(_TO_HOP_BY_CODE.get(code, [])) or None,
                "diem_chuan": thi,
                "_source_file": "diem_chuan_2025_full.html",
                "_row_index": row_index,
            })
        else:
            skipped.append({
                "row_index": row_index,
                "reason": "no_thi_score",
                "code": code,
                "raw_row": r,
            })
        if hb is not None:
            emitted.append({
                "major_code": code,
                "major_name": _code_to_name(code),
                "year": 2025,
                "method": "hoc_ba",
                "to_hop": list(_TO_HOP_BY_CODE.get(code, [])) or None,
                "diem_chuan": hb,
                "_source_file": "diem_chuan_2025_full.html",
                "_row_index": row_index,
            })
        else:
            skipped.append({
                "row_index": row_index,
                "reason": "no_hb_score",
                "code": code,
                "raw_row": r,
            })
    return emitted, skipped


def _parse_2024(html: str, method: str) -> Tuple[List[dict], List[dict], bool]:
    """Parse a 2024 file.

    The 2024 HTML files (per inspection) have only:
      - an <h3> title naming year + method
      - an <img> with the actual table (PNG collage — NOT parseable as HTML)
    So we return ZERO rows but flag a warning. The recommender will still
    work on 2025-only data; the user explicitly said "abstain beats fabricating".
    """
    rows = _extract_table_rows(html)
    if not rows:
        # Image-only / no tabular data: honest skip, logged.
        return [], [{
            "reason": "no_tabular_data",
            "detail": "2024 file contains only an <img> of the table; not parseable as HTML",
            "method": method,
        }], True
    # Defensive: if SOMEHOW a future crawl embeds a real <table>, attempt a
    # generic best-effort parse. (We do NOT fabricate scores; we only
    # attempt to read whatever is there.)
    emitted: List[dict] = []
    skipped: List[dict] = []
    row_index = 0
    for r in rows:
        if len(r) < 4:
            continue
        joined = " ".join(r).lower()
        if "số tt" in joined or "mã ngành" in joined or "tên trường" in joined:
            continue
        # Locate the major code cell
        code_cell = next((c for c in r if _looks_like_major_code(c)), None)
        if not code_cell:
            skipped.append({"row_index": row_index, "reason": "no_major_code", "raw_row": r})
            row_index += 1
            continue
        code = code_cell.strip()
        # Pick the FIRST cell that parses as a thang-30 score (0..30)
        score = None
        for c in r:
            v = _to_float(c)
            if v is not None and 0 <= v <= 30:
                score = v
                break
        if score is None:
            skipped.append({"row_index": row_index, "reason": "no_in_scale_score", "code": code, "raw_row": r})
            row_index += 1
            continue
        emitted.append({
            "major_code": code,
            "major_name": _code_to_name(code),
            "year": 2024,
            "method": method,
            "to_hop": list(_TO_HOP_BY_CODE.get(code, [])) or None,
            "diem_chuan": score,
            "_source_file": f"diem_chuan_2024_{method.replace('thi_thpt','thi').replace('hoc_ba','hb')}.html",
            "_row_index": row_index,
        })
        row_index += 1
    return emitted, skipped, False


# ---- Top-level entrypoint ---------------------------------------------------

def parse_all(log_skipped: bool = True) -> Dict[str, Any]:
    """Parse every configured source, build the dedup history + log.

    Dedup rule: (major_code, year, method) is unique. If the SAME triple
    appears in multiple files, the FIRST emitted wins (2025 is iterated
    first, then 2024).
    """
    _load_name_maps()
    whitelist: Set[str] = set(get_whitelist())

    history: Dict[Tuple[str, int, str], dict] = {}
    skipped: List[dict] = []
    file_reports: List[dict] = []

    for fname, year, method in _SOURCES:
        path = _RAW_DIR / fname
        if not path.exists():
            file_reports.append({"file": fname, "status": "missing"})
            continue
        html = path.read_text(encoding="utf-8", errors="ignore")
        if fname == "diem_chuan_2025_full.html":
            emitted, file_skipped = _parse_2025_full(html)
        else:
            emitted, file_skipped, _ = _parse_2024(html, method)
        file_reports.append({
            "file": fname,
            "year": year,
            "method": method,
            "emitted": len(emitted),
            "skipped": len(file_skipped),
        })
        for row in emitted:
            key = (row["major_code"], row["year"], row["method"])
            if key in history:
                continue
            # Apply whitelist filter (with log on the row)
            if row["major_code"] not in whitelist:
                skipped.append({
                    "reason": "not_in_2026_whitelist",
                    "code": row["major_code"],
                    "name": row["major_name"],
                    "year": year,
                    "method": method,
                })
                continue
            history[key] = row
        # log source-level skips (e.g. "no tabular data")
        skipped.extend(file_skipped)

    # Final: sort by (year DESC, major_code, method) for stable diffs.
    entries = sorted(
        history.values(),
        key=lambda e: (-e["year"], e["major_code"], e["method"]),
    )
    doc: Dict[str, Any] = {
        "_meta": {
            "schema": "phase_a_v1",
            "generated_at": _now_iso(),
            "sources": file_reports,
            "whitelist_size": len(whitelist),
            "skipped_count": len(skipped),
            "note": (
                "2024 HTML files contain only an <img> of the table (no parseable HTML "
                "rows). 2025 is the only year with tabular data; the recommender falls "
                "back to a single-year view (still honest, basis+disclaimer surface it)."
            ),
        },
        "entries": entries,
    }
    if log_skipped:
        doc["_meta"]["skipped"] = skipped[:200]  # cap to keep file small
    return doc


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_history(out_path: Path = _OUT_PATH) -> Path:
    doc = parse_all()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Parse diem_chuan history HTML → JSON")
    p.add_argument("--out", default=str(_OUT_PATH), help="Output JSON path")
    args = p.parse_args(argv)
    out = write_history(Path(args.out))
    doc = json.loads(out.read_text(encoding="utf-8"))
    print(f"[parse_diem_chuan_history] wrote {out}")
    print(f"[parse_diem_chuan_history] entries: {len(doc['entries'])}")
    print(f"[parse_diem_chuan_history] skipped: {doc['_meta'].get('skipped_count', 0)}")
    for r in doc["_meta"]["sources"]:
        print(f"  - {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
