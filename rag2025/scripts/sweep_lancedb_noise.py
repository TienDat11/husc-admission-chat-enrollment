# @spec(S9) LanceDB read-only noise sweep
# Quantifies stale-2025, low-quality, duplicates, and embedding-dim drift.
# CRITICAL: this script is strictly READ-ONLY — it must NEVER call
# add / create_table / delete / merge_insert / open_table(..., mode="overwrite").

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb
from loguru import logger

# NOTE: settings is imported lazily inside _settings_dim() to avoid instantiating
# RAGSettings() at module-load time. RAGSettings reads .env + env vars and can
# race with pytest fixtures that patch sys.modules; deferring the import keeps
# the module importable in isolation (and in test reload cycles).



# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

# A row is "low quality" if any of:
#   - its text is shorter than 100 chars
#   - it starts with "MỤC LỤC" (case-insensitive) — a table-of-contents marker
#   - it is mostly dots/digits — a TOC / page-number pattern
_SHORT_TEXT_THRESHOLD = 100
_TOC_HEADER_RE = re.compile(r"^\s*M[ỤU]C\s+L[ỤU]C\b", re.IGNORECASE)
# A row qualifies as "mostly dots/digits" if >= 40% of its non-whitespace chars
# are digits, dots, or semicolons (typical TOC/page-number strings).
_DOTS_DIGITS_RATIO = 0.40


def _is_low_quality(text: str) -> bool:
    if not text:
        return True
    if len(text) < _SHORT_TEXT_THRESHOLD:
        return True
    if _TOC_HEADER_RE.match(text):
        return True
    # Density check on non-whitespace chars
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return True
    dot_or_digit = sum(1 for c in compact if c.isdigit() or c in ".;")
    if len(compact) >= 30 and (dot_or_digit / len(compact)) >= _DOTS_DIGITS_RATIO:
        return True
    return False


def _normalize_text(text: str) -> str:
    """Lowercase + collapse all whitespace runs to a single space."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


# ---------------------------------------------------------------------------
# Row parsing
# ---------------------------------------------------------------------------

def _parse_metadata(raw: Any) -> dict:
    """Parse the metadata_json column. Returns {} on any failure (graceful)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


def _is_truthy_superseded(value: Any) -> bool:
    """Treat None, False, 0, "", and missing keys as falsy."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def _load_settings():
    """Import the settings singleton, tolerating both run contexts.

    Works whether the repo root (`rag2025.config.settings`) or the package dir
    (`config.settings`) is on sys.path — the real CLI runs from inside rag2025/
    so the bare `config.settings` form is the one that resolves there. We make
    sure the rag2025 package dir (this file's parent.parent) is importable.
    """
    _pkg_root = str(Path(__file__).resolve().parent.parent)
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    try:
        from rag2025.config.settings import settings
        return settings
    except Exception:
        from config.settings import settings  # run-from-inside-rag2025
        return settings


def _lancedb_uri() -> str:
    """Lazy read of LANCEDB_URI. Honors test override."""
    if _LANCEDB_URI_OVERRIDE is not None:
        return str(_LANCEDB_URI_OVERRIDE)
    try:
        return str(getattr(_load_settings(), "LANCEDB_URI", "./data/lancedb"))
    except Exception:  # pragma: no cover — defensive
        return "./data/lancedb"


def _lancedb_table_name() -> str:
    """Lazy read of LANCEDB_TABLE. Honors test override."""
    if _LANCEDB_TABLE_OVERRIDE is not None:
        return str(_LANCEDB_TABLE_OVERRIDE)
    try:
        return str(getattr(_load_settings(), "LANCEDB_TABLE", "rag2025"))
    except Exception:  # pragma: no cover — defensive
        return "rag2025"


def _settings_dim() -> int:
    """Lazy import + read of settings.EMBEDDING_DIM.

    Defers RAGSettings() instantiation to first call (and only if not already
    cached). Tests can patch either rag2025.config.settings.settings.EMBEDDING_DIM
    or the module-level _SETTINGS_DIM_OVERRIDE attribute to force a value.
    """
    if _SETTINGS_DIM_OVERRIDE is not None:
        return int(_SETTINGS_DIM_OVERRIDE)
    try:
        return int(getattr(_load_settings(), "EMBEDDING_DIM", 0))
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning("Could not read settings.EMBEDDING_DIM ({}); defaulting to 0", exc)
        return 0


# Module-level override hook used by tests (and CLI flags) to inject a specific
# EMBEDDING_DIM without mutating the global settings instance. When None, the
# real settings.EMBEDDING_DIM is used.
_SETTINGS_DIM_OVERRIDE: int | None = None
_LANCEDB_URI_OVERRIDE: str | None = None
_LANCEDB_TABLE_OVERRIDE: str | None = None


def _read_rows(table) -> list[dict]:
    """Read all rows from a LanceDB table — READ-ONLY path.

    Prefers to_arrow().to_pylist() (matches the existing audit script). Falls
    back to to_pandas().to_dict('records') if to_arrow is unavailable.
    """
    to_arrow = getattr(table, "to_arrow", None)
    if to_arrow is not None:
        try:
            return list(to_arrow().to_pylist())
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("to_arrow() failed ({}); falling back to to_pandas()", exc)

    to_pandas = getattr(table, "to_pandas", None)
    if to_pandas is not None:
        df = to_pandas()
        if df is not None:
            return df.to_dict("records")

    raise RuntimeError("Table exposes neither to_arrow() nor to_pandas() — cannot read rows")


def sweep(table) -> dict[str, Any]:
    """Quantify noise categories in a LanceDB table.

    Args:
        table: an already-opened LanceDB table object. Must be read-only.

    Returns:
        A dict with the following keys:
          - sweep_date (ISO 8601 UTC)
          - total_chunks
          - stale_2025_without_superseded
          - low_quality_count
          - low_quality_pct
          - duplicate_text_count
          - duplicate_pct
          - dim_drift: {stored_dim, settings_dim, mismatch}
          - by_year: {year_str: count, ...}
    """
    rows = _read_rows(table)
    total = len(rows)

    by_year: Counter = Counter()
    stale_2025 = 0
    low_quality = 0
    normalized_texts: list[str] = []

    for row in rows:
        metadata = _parse_metadata(row.get("metadata_json"))

        data_year = metadata.get("data_year")
        is_superseded = _is_truthy_superseded(metadata.get("is_superseded"))

        # Year aggregation (count missing/None as "null")
        year_key = str(data_year) if data_year is not None else "null"
        by_year[year_key] += 1

        # Stale 2025: data_year == 2025 AND is_superseded is falsy
        if data_year == 2025 and not is_superseded:
            stale_2025 += 1

        # Low quality on the text column
        text = row.get("text") or ""
        if _is_low_quality(text):
            low_quality += 1

        # Normalize text for duplicate detection
        normalized_texts.append(_normalize_text(text))

    unique_norm = set(normalized_texts)
    duplicate_count = max(0, total - len(unique_norm))

    # Embedding-dim drift: compare first row's vector to settings.EMBEDDING_DIM
    stored_dim: int | None = None
    if rows:
        first_vector = rows[0].get("vector")
        if first_vector is not None:
            try:
                stored_dim = len(first_vector)
            except TypeError:
                stored_dim = None
    settings_dim = _settings_dim()
    mismatch = (stored_dim is not None) and (stored_dim != settings_dim)

    return {
        "sweep_date": datetime.now(timezone.utc).isoformat(),
        "total_chunks": total,
        "stale_2025_without_superseded": stale_2025,
        "low_quality_count": low_quality,
        "low_quality_pct": round(low_quality / total, 6) if total else 0.0,
        "duplicate_text_count": duplicate_count,
        "duplicate_pct": round(duplicate_count / total, 6) if total else 0.0,
        "dim_drift": {
            "stored_dim": stored_dim,
            "settings_dim": settings_dim,
            "mismatch": bool(mismatch),
        },
        "by_year": dict(by_year),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_sweep(report: dict, output_path: Path) -> None:
    """Write the sweep report dict to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# Anchor output to rag2025/results/... relative to THIS file, so it lands in the
# same place whether the CLI is invoked from the repo root or from inside rag2025/.
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "results" / "eval_harness" / "lancedb_noise_report.json"


def main(argv: list[str] | None = None, output_path: Path | None = None) -> int:
    """Open the real LanceDB table READ-ONLY, run sweep(), and persist the report.

    This function NEVER calls add / create_table / delete / merge_insert.
    It opens the table with lancedb.connect(uri) followed by db.open_table(name)
    (no mode="overwrite" is ever passed).

    Args:
        argv: optional argv list (defaults to sys.argv[1:]). Tests pass an
              empty list to skip argparse parsing of pytest's sys.argv.
        output_path: optional override for the JSON output path. When provided,
                     argparse's --output is ignored.
    """
    parser = argparse.ArgumentParser(
        description="Read-only LanceDB noise sweep (stale / low-quality / dupes / dim-drift)"
    )
    parser.add_argument(
        "--lancedb-uri",
        default=_lancedb_uri(),
        help="LanceDB URI (default: settings.LANCEDB_URI)",
    )
    parser.add_argument(
        "--table",
        default=_lancedb_table_name(),
        help="Table name (default: settings.LANCEDB_TABLE)",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output JSON path (default: rag2025/results/eval_harness/lancedb_noise_report.json)",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    target = Path(output_path) if output_path is not None else Path(args.output)

    try:
        logger.info("Sweeping table {!r} at {!r} (READ-ONLY)", args.table, args.lancedb_uri)

        # --- READ-ONLY: connect + open_table only. NO create_table. NO mode="overwrite". ---
        db = lancedb.connect(args.lancedb_uri)
        table = db.open_table(args.table)

        report = sweep(table)
        report["table"] = args.table
        report["lancedb_uri"] = args.lancedb_uri

        save_sweep(report, target)
        logger.info("Sweep report saved to {!r}", target)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    except Exception:
        logger.exception("Sweep failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
