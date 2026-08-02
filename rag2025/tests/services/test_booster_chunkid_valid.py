# @spec(AUDIT-FIX) test_booster_chunkid_valid — BUG-A guard
"""TDD guard: every chunk_id referenced by ``_AGGREGATION_RULES`` MUST
exist in the LanceDB ``rag2025`` table once ``{year}`` is expanded to
the current admission year. This is the regression guard for BUG-A
(booster rules referenced dead ``husc_thongbao_idNN_{year}`` chunk_ids
that silently failed injection).

Scope:
  - Load ``_AGGREGATION_RULES`` (year-templated).
  - Expand ``{year}`` → current admission year (2026 in prod, but the
    test pins env to 2026 for determinism).
  - For every chunk_id in every rule's ``chunks`` list, assert it
    exists in the ``rag2025`` LanceDB table.

Failure mode pinned: a stale/renamed chunk_id in the booster rules
makes ``fetch_by_id`` return None and the booster silently de-boosts.
This test catches that BEFORE production.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

RAG_ROOT = Path(__file__).resolve().parents[2]


def _all_rule_chunk_ids(year: int) -> set:
    """Expand every rule's chunk_id list at the given year."""
    import importlib
    from src.services import temporal_authority, aggregation_rules
    importlib.reload(temporal_authority)
    importlib.reload(aggregation_rules)
    ids = set()
    for rule in aggregation_rules._AGGREGATION_RULES:
        for cid in rule["chunks"]:
            try:
                ids.add(cid.format(year=year))
            except (KeyError, IndexError):
                ids.add(cid)
    return ids


def test_all_booster_chunk_ids_exist_in_db(monkeypatch) -> None:
    """Every chunk_id ref'd by booster rules must exist in rag2025 table.

    This is the BUG-A regression guard: if a rule points at a
    non-existent chunk_id (e.g. ``husc_thongbao_id49_2026`` after the
    chunk was renamed), fetch_by_id silently returns None and the
    booster de-boosts without any error. We fail loudly here instead.
    """
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")
    monkeypatch.delenv("APP_ENV", raising=False)

    import importlib
    from src.services import temporal_authority
    importlib.reload(temporal_authority)
    year = int(temporal_authority.get_current_admission_year())
    assert year == 2026, f"Test setup error: expected year=2026, got {year}"

    refs = _all_rule_chunk_ids(year)
    assert refs, "Booster rules produced no chunk_ids — test setup error"

    # Load LanceDB table once and check membership.
    import lancedb
    settings_path = RAG_ROOT / ".env"
    if settings_path.exists():
        from dotenv import load_dotenv
        load_dotenv(settings_path)
    from config.settings import RAGSettings
    settings = RAGSettings()

    db = lancedb.connect(settings.LANCEDB_URI)
    table = db.open_table(settings.LANCEDB_TABLE)
    df = table.to_pandas()
    db_ids = set(str(v) for v in df["id"].tolist())

    stale = sorted(cid for cid in refs if cid not in db_ids)
    assert not stale, (
        f"BUG-A regression: {len(stale)} booster chunk_ids do NOT exist "
        f"in LanceDB table {settings.LANCEDB_TABLE!r}: {stale}"
    )


def test_all_booster_chunk_ids_exist_at_year_2027(monkeypatch) -> None:
    """Year-rollover guard: at clock=2027 every _2027-suffixed chunk_id
    must also exist (i.e. rules must not hardcode dead 2026 ids that
    silently fail after year rollover). We skip if the 2027 chunks are
    not yet ingested (env CI without future data) — but we DO assert
    that NO rule references a hardcoded ``_2026`` literal that would
    bypass the {year} templating."""
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2027")
    monkeypatch.delenv("APP_ENV", raising=False)

    import importlib
    from src.services import temporal_authority, aggregation_rules
    importlib.reload(temporal_authority)
    importlib.reload(aggregation_rules)
    year = int(temporal_authority.get_current_admission_year())
    assert year == 2027

    # Hardcoded 2026 literal in any rule's chunks is a BUG (defeats
    # year templating). Allowlist: ``so_sanh_hocphi_2025_vs_2026`` is an
    # intentional historical-vs-current comparison chunk that is BY
    # DESIGN pinned to the 2025→2026 transition (the only year pair we
    # publish a tuition-delta for).
    ALLOWLIST_2026_LITERALS = {"so_sanh_hocphi_2025_vs_2026"}
    hardcoded_2026 = []
    for rule in aggregation_rules._AGGREGATION_RULES:
        for cid in rule["chunks"]:
            if "_2026" in cid and "{year}" not in cid and cid not in ALLOWLIST_2026_LITERALS:
                hardcoded_2026.append(cid)
    assert not hardcoded_2026, (
        f"Booster rules contain hardcoded _2026 chunk_ids (should use "
        f"{{year}} template): {hardcoded_2026}"
    )
