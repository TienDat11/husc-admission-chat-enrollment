"""
Retrieval Layer - LanceDB Interface

Fail-safe dense retrieval for embedded LanceDB backend.
Includes temporal filtering: auto-prefer newest year, exclude superseded.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from config.settings import RAGSettings
from src.infrastructure.lancedb_adapter import LanceDBAdapter
from services._metadata_helpers import get_legacy_source
# Server-side tombstone gate, used by the dense arm (adapter.search) when the
# table carries the `is_tombstone` column. Verified against lancedb 0.30.2:
# a bare `is_tombstone = false` EXCLUDES NULL values (3-valued-logic), so
# the explicit `OR IS NULL` keeps un-stamped rows in the result set.
#
# SCHEMA GATE (T15 regression fix): legacy tables (the `rag2025` fixture,
# the chaos-drill publish/rollback tables) predate this column and raise
# LanceError(Schema) if it is referenced. `retrieve()` consults
# `adapter.has_tombstone_column()` and passes `where_clause=None` when the
# column is absent — the post-fetch belt in `_apply_temporal_filter` still
# covers the metadata-mirror stamping convention in that case.
TOMBSTONE_PREDICATE = "(is_tombstone = false OR is_tombstone IS NULL)"


def _current_as_of_int() -> int:
    """Today as a YYYYMMDD int on the temporal authority's testable clock."""
    from services.temporal_authority import _now
    d = _now().date()
    return d.year * 10000 + d.month * 100 + d.day


def _drop_by_effectiveness(row: Dict[str, Any], meta: Dict[str, Any],
                           as_of: int) -> bool:
    """T15 effectiveness law (D1 R03): a served row is current and
    inside its effective window.

    Belt-level (post-fetch), schema-gated by NULL: every writer stamps
    flat `status="current"` and `effective_from=0`/`effective_to=99991231`,
    so a NULL/absent column means an un-stamped legacy row → KEEP.

    Flat `status`: drop iff present AND != "current". Metadata `status`:
    explicit-deny (draft/review/approved/archived) — legacy rows may carry an
    unrelated `status` key, so served-set membership would drop them.
    """
    _status = row.get("status")
    if _status is not None and _status != "current":
        return True
    _meta_status = meta.get("status")
    if _meta_status in ("draft", "review", "approved", "archived"):
        return True
    _ef = row.get("effective_from")
    if isinstance(_ef, float) and math.isnan(_ef):
        _ef = None
    _et = row.get("effective_to")
    if isinstance(_et, float) and math.isnan(_et):
        _et = None
    if _ef is not None and _ef > as_of:
        return True
    if _et is not None and _et < as_of:
        return True
    return False


# --- Temporal Configuration ---
def _detect_current_admission_year() -> int:
    """Detect current admission year from LanceDB chunks or fallback to clock.

    Scans chunks for the highest data_year, so when 2027 data arrives
    the system automatically treats 2027 as current without config changes.
    """
    try:
        from src.infrastructure.lancedb_adapter import LanceDBAdapter
        settings = RAGSettings()
        adapter = LanceDBAdapter(uri=settings.LANCEDB_URI, table_name=settings.LANCEDB_TABLE)
        if adapter.table_exists():
            adapter.ensure_table()
            table = adapter._table
            if "data_year" in table.schema.names:
                # `search().select()` pushes the projection into Lance. Plain
                # `to_arrow()` materializes the 1024-dim vector column first —
                # measured 18ms/+9.4MB vs 6ms/+0.0MB on 2138 rows, and this runs
                # at module import. `to_pandas(columns=)` does not exist on 0.29.2.
                df = table.search().select(["data_year"]).to_pandas()
                max_year = int(df["data_year"].max())
                if max_year >= 2025:
                    return max_year
    except Exception:
        pass
    # Fallback: use system clock year (admissions happen mid-year)
    now = datetime.now()
    return now.year if now.month >= 1 else now.year - 1


CURRENT_ADMISSION_YEAR = _detect_current_admission_year()
YEAR_BOOST_FACTOR = 1.5       # 50% boost for current-year chunks
OLD_YEAR_DEMOTE_FACTOR = 0.7  # 30% demote for non-current-year chunks
# G1-T4 (durability S14.x): match any 4-digit year literal (20\d{2}) so
# queries like "so sánh 2025 vs 2026" classify as historical in 2028 too.
# The original enumerated list (2024|2023|2022|2021) silently dropped
# newer historical references like 2025 / 2026 once they fell out of the
# 4-year window.
HISTORICAL_QUERY_PATTERN = re.compile(
    r"\b(so sánh|so sanh|đối chiếu|doi chieu|thay đổi|thay doi|tăng|giảm|so với|so voi|năm\s+\b20\d{2}\b|năm\s+(trước đó|năm trước)|trước|cũ|historical|vs|năm\s+20\d{2})\b",
    re.IGNORECASE,
)


def _pick_promoted_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deterministically pick ONE row when a logical doc_id matches several.

    Ticket 45 (Loi B): after the canonical ingest, several physical rows
    can share one ``doc_id`` (measured: up to 3 per doc_id on the live
    table). Resolving a symbolic booster id must therefore be a
    *function of the row set*, not of LanceDB scan order — reordering the
    input must return the same winner.

    Promotion rule (documented, in order):
      1. Non-``legacy`` release wins over ``legacy`` (a promoted canonical
         row supersedes the legacy-ingest duplicate).
      2. Largest ``data_year`` wins (newest admission cycle).
      3. Smallest ``id`` lexicographically breaks any remaining tie
         (stable across runs; no reliance on storage order).
    """
    def key(row: Dict[str, Any]) -> tuple:
        try:
            year = int(row.get("data_year") or 0)
        except (TypeError, ValueError):
            year = 0
        return (str(row.get("release_id") or "") == "legacy", -year, str(row.get("id") or ""))

    return min(rows, key=key)


class RetrievalError(Enum):
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    EMPTY_RESULT = "empty_result"
    INVALID_PAYLOAD = "invalid_payload"
    UNKNOWN_ERROR = "unknown_error"


@dataclass(frozen=True)
class RetrievedDocument:
    text: str
    source: Optional[str] = None
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    point_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "metadata": self.metadata,
            "score": self.score,
        }


@dataclass
class RetrievalResult:
    documents: List[RetrievedDocument] = field(default_factory=list)
    error_type: Optional[RetrievalError] = None
    error_message: Optional[str] = None
    confidence: float = 0.0
    degraded_filter: bool = False  # M3-N4b: True when prior-year fallback rows returned

    @property
    def is_success(self) -> bool:
        return self.error_type is None and len(self.documents) > 0


class LanceDBRetrieverConfig:
    def __init__(
        self,
        uri: str,
        table_name: str,
        embedding_dim: int = 4096,
        default_top_k: int = 5,
    ):
        self.uri = uri
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.default_top_k = default_top_k


class LanceDBRetriever:
    def __init__(self, config: LanceDBRetrieverConfig) -> None:
        self._config = config
        self._adapter = LanceDBAdapter(uri=config.uri, table_name=config.table_name)
        self._adapter.ensure_table()

    @classmethod
    def from_env(cls) -> "LanceDBRetriever":
        settings = RAGSettings()
        config = LanceDBRetrieverConfig(
            uri=settings.LANCEDB_URI,
            table_name=settings.LANCEDB_TABLE,
            embedding_dim=settings.EMBEDDING_DIM,
            default_top_k=5,
        )
        return cls(config=config)

    def retrieve(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
    ) -> RetrievalResult:
        if not query_vector:
            return RetrievalResult(
                error_type=RetrievalError.INVALID_PAYLOAD,
                error_message="Query vector is empty",
            )

        if len(query_vector) != self._config.embedding_dim:
            return RetrievalResult(
                error_type=RetrievalError.INVALID_PAYLOAD,
                error_message=(
                    f"Query vector dimension mismatch: "
                    f"expected {self._config.embedding_dim}, got {len(query_vector)}"
                ),
            )

        limit = top_k or self._config.default_top_k

        # S14.3 — temporal-intent-driven POST-fetch hard-EXCLUDE.
        #
        # The real LanceDB table has NO top-level `data_year` column (the
        # year lives inside the `metadata_json` JSON-string column), so a
        # SQL `where_clause = "data_year = N OR data_year IS NULL"` passed
        # to `adapter.search` raises LanceError(Schema). Per architect
        # decision we keep the schema as-is and instead apply the hard
        # year-filter AFTER the adapter returns rows, in
        # `_apply_temporal_filter`, which already parses `metadata_json`
        # per row. The over-fetch headroom `top_k = max(limit*5, limit)`
        # gives enough room to drop prior-year rows and still return
        # `limit` current-year / year-agnostic stable chunks.
        #
        # The `where_clause` passed to `adapter.search` is the tombstone
        # gate (T15): it excludes rows stamped `is_tombstone = true`
        # server-side and keeps un-stamped (NULL) rows. It is NOT a
        # `data_year` clause — that remains forbidden. The belt is the
        # hard-drop in `_apply_temporal_filter` below.
        #
        # Schema gate: legacy tables predate the `is_tombstone` column and
        # raise `LanceError(Schema)` when it is referenced; when the
        # column is absent, `where_clause=None` and the post-fetch belt
        # (metadata-mirror stamp) still excludes tombstoned rows.
        try:
            tombstone_gate = (
                TOMBSTONE_PREDICATE
                if self._adapter.has_tombstone_column()
                else None
            )
        except Exception:
            # A schema probe error must not crash retrieval: fail safe to
            # no server-side gate (post-fetch belt covers the stamp).
            tombstone_gate = None
        try:
            rows = self._adapter.search(
                query_vector=query_vector,
                top_k=max(limit * 5, limit),
                where_clause=tombstone_gate,
            )
            rows = self.apply_post_fetch_filters(
                rows, metadata_filter, query=query
            )
            docs = self._normalize_rows(rows[:limit])
            if not docs:
                return RetrievalResult(
                    documents=[],
                    error_type=RetrievalError.EMPTY_RESULT,
                    error_message="No matching documents found",
                    confidence=0.0,
                )
            # M3-N4b: aggregate degraded_filter flag
            degraded = any(d.metadata.get("degraded_filter") for d in docs)
            return RetrievalResult(documents=docs, confidence=self._calculate_confidence(docs), degraded_filter=degraded)
        except Exception as exc:
            logger.error(f"LanceDB retrieval failed: {exc}")
            return RetrievalResult(
                error_type=RetrievalError.UNKNOWN_ERROR,
                error_message=str(exc),
            )

    def _apply_temporal_filter(
        self,
        rows: List[Dict[str, Any]],
        metadata_filter: Optional[Dict[str, Any]],
        temporal_intent: Any = None,
        admission_ctx: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Apply temporal scoring: boost current-year, demote old, exclude superseded.

        When `temporal_intent` AND `admission_ctx` are both supplied (the
        S14.3 post-fetch path), this function also performs a hard-EXCLUDE
        of prior-year rows in the IN_SEASON+current/ambiguous case — the
        replacement for the broken SQL `data_year = N` clause (the real
        table has no top-level `data_year` column).

        Hard-EXCLUDE rules (per architect decision, replacing SQL pre-filter):
          * season == IN_SEASON AND intent in {current, ambiguous}
              → DROP rows whose parsed `data_year` is present AND `< current_year`
                (i.e. prior-year chunks). KEEP `data_year == current_year` AND
                `data_year is None/missing` (year-agnostic stable chunks —
                the post-fetch equivalent of the OR-NULL rule).
              → also DROP superseded as today.
          * season == IN_SEASON AND intent == historical → no hard drop; keep
            prior years (soft demote).
          * season == IN_SEASON AND intent == cross_year → no hard drop; keep
            all years.
          * season == PRE_SEASON_GAP (or OFF_SEASON) → no hard drop; show
            prior years but apply soft demote so current (if any) ranks first.

        Backward-compat: when `temporal_intent` is None (legacy path —
        caller did not pass `query`), behave EXACTLY as before (soft
        demote only, no hard drop). Existing callers that don't pass
        `query` must be unaffected.

        Rules (from Phase 6 temporal strategy):
        1. Same-topic, newer year boosts: score * YEAR_BOOST_FACTOR for current year
        2. Superseded chunks excluded by default (unless query asks for historical data)
        3. Non-current-year chunks demoted by OLD_YEAR_DEMOTE_FACTOR
        4. When year is explicitly queried (e.g. "năm 2025"), include that year's chunks

        IMPORTANT (S14.2): the year anchor is resolved PER CALL via
        `temporal_authority.get_current_admission_year()`. The module-level
        `CURRENT_ADMISSION_YEAR` constant is only a best-effort initial value
        for legacy importers — it is NOT the authority.
        """
        if not rows:
            return rows

        # Per-call year anchor (S14.2 import-freeze fix). Lazy import to avoid
        # order-of-import issues — mirrors the pattern in
        # `services.major_code_validator.get_whitelist()`.
        from services.temporal_authority import get_current_admission_year
        from services.season import SeasonPhase
        from services.temporal_intent import TemporalIntent
        current_year = int(get_current_admission_year())
        as_of = _current_as_of_int()

        # Determine whether the hard-EXCLUDE path is active.
        # Both temporal_intent AND admission_ctx must be supplied. If only
        # one is present, fall back to legacy soft path (defensive).
        hard_exclude_active = (
            temporal_intent is not None
            and admission_ctx is not None
        )
        season = admission_ctx.get("season") if hard_exclude_active else None
        has_data = bool(admission_ctx.get("has_data")) if hard_exclude_active else False

        # Detect if query explicitly asks for a specific year
        query_year = None
        is_historical_query = False
        if metadata_filter:
            query_year = metadata_filter.get("data_year") or metadata_filter.get("year")
            # Check if filter indicates a historical query
            or_conds = metadata_filter.get("or_conditions", [])
            for cond in or_conds:
                if cond.get("data_year") or cond.get("year"):
                    query_year = query_year or cond.get("data_year") or cond.get("year")

        # Also check the raw query string for historical intent
        query_text = ""
        if metadata_filter:
            query_text = metadata_filter.get("query", "")
        if query_text and HISTORICAL_QUERY_PATTERN.search(query_text):
            is_historical_query = True

        # Decide whether to apply the prior-year hard-EXCLUDE for THIS call.
        # IN_SEASON + (current OR ambiguous) is the only path that drops
        # prior-year rows. historical / cross_year / PRE_SEASON_GAP keep them.
        hard_drop_prior_year = bool(
            hard_exclude_active
            and has_data
            and season == SeasonPhase.IN_SEASON
            and temporal_intent in (TemporalIntent.current, TemporalIntent.ambiguous)
        )

        kept_rows: List[Dict[str, Any]] = []
        for row in rows:
            raw_meta = row.get("metadata_json") or row.get("metadata") or "{}"
            meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
            # T15: a tombstoned row is NEVER valid — no historical-query
            # exemption, no legacy soft path. Check the top-level column
            # first, then the metadata mirror, so either stamping
            # convention is caught. (The server-side predicate in
            # `adapter.search` already excludes these rows; this is the
            # belt that covers any caller that bypasses it.)
            is_tombstone = row.get("is_tombstone")
            if is_tombstone is None:
                is_tombstone = meta.get("is_tombstone")
            if is_tombstone:
                continue

            # T15 effectiveness law (D1 R03): status + effective window.
            # Unconditional (not gated on hard_exclude_active) so the legacy
            # soft path (query=None) obeys the same law. NULL/absent = KEEP.
            if _drop_by_effectiveness(row, meta, as_of):
                continue

            # Rule 2: Exclude superseded chunks (unless historical query)
            is_superseded = meta.get("is_superseded", False)
            if is_superseded and not is_historical_query:
                # Hard-drop (or push far down). For the hard-EXCLUDE path
                # we drop outright; for the soft path we demote with +100.
                if hard_exclude_active:
                    continue
                row["_distance"] = row.get("_distance", 999.0) + 100.0  # Push far down
                row["_temporal_demoted"] = True
                kept_rows.append(row)  # Keep demoted superseded rows in legacy soft path
                continue

            # Determine chunk's data year
            chunk_year = meta.get("data_year") or meta.get("year")
            if isinstance(chunk_year, str):
                try:
                    chunk_year = int(chunk_year)
                except (ValueError, TypeError):
                    chunk_year = None

            # Hard-EXCLUDE: drop prior-year rows in IN_SEASON+current/ambiguous.
            # A missing data_year (chunk_year is None) is a year-agnostic stable
            # chunk and MUST survive (the post-fetch equivalent of the
            # `OR data_year IS NULL` rule). A row whose data_year matches the
            # current year also survives.
            if hard_drop_prior_year:
                if chunk_year is not None and isinstance(chunk_year, int) and chunk_year < current_year:
                    # Tổ hợp overview chunks are year-agnostic mapping data —
                    # preserve them regardless of year (BUG-6 fix).
                    _cid = row.get("chunk_id") or row.get("id") or ""
                    if isinstance(_cid, str) and _cid.startswith("qa_chi_tieu_khoi_xet_"):
                        kept_rows.append(row)
                        # Apply temporal demote (soft) instead of hard-drop
                        distance = row.get("_distance", 0.0)
                        row["_distance"] = distance / OLD_YEAR_DEMOTE_FACTOR
                        row["_temporal_demoted"] = True
                        continue
                    # Prior-year chunk — hard-drop.
                    continue
                # else: keep (current-year OR year-agnostic None OR non-int year-tag)

            # Rule 4: If query explicitly asks for a year, respect it
            if query_year and chunk_year == query_year:
                kept_rows.append(row)
                continue  # No boost/demote, let similarity score decide

            # Rule 1 & 3: Boost current year, demote others (per-call anchor)
            if chunk_year == current_year:
                distance = row.get("_distance", 0.0)
                if distance > 0:
                    row["_distance"] = distance / YEAR_BOOST_FACTOR  # Lower distance = higher score
                row["_temporal_boosted"] = True
            elif chunk_year and chunk_year < current_year:
                distance = row.get("_distance", 0.0)
                row["_distance"] = distance / OLD_YEAR_DEMOTE_FACTOR  # Higher distance = lower score
                row["_temporal_demoted"] = True
                # M3-N4b: prior-year kept via fallback → flag degraded
                row["_degraded_filter"] = True

            # M3-N4b: is_reference flat column (or metadata fallback) → demote + flag, never drop
            is_ref = row.get("is_reference")
            if is_ref is None:
                is_ref = meta.get("is_reference")
            if is_ref and not row.get("_degraded_filter"):
                distance = row.get("_distance", 0.0)
                row["_distance"] = distance / OLD_YEAR_DEMOTE_FACTOR
                row["_temporal_demoted"] = True
                row["_degraded_filter"] = True

            kept_rows.append(row)

        # Re-sort by adjusted distance (lower is better)
        kept_rows.sort(key=lambda r: r.get("_distance", 999.0))
        return kept_rows

    # ------------------------------------------------------------------
    # S14.2b — data-presence probe + admission context (per-call)
    # ------------------------------------------------------------------
    def has_current_year_data(self, year: int) -> bool:
        """Cheap probe: does the index contain at least one chunk with
        `data_year == year`?

        Implementation notes (per S14.2b plan):
          * Caches the distinct-year SET for process lifetime, keyed by table
            row-count. When the row-count changes (e.g., after an ingest), the
            cache is refreshed automatically.
          * Conservative default: on ANY error, returns True. We prefer the
            normal retrieval path over accidentally triggering
            PRE_SEASON_GAP / empty-answer gap behavior in production. The
            empty-result path already handles the truly-empty case downstream.
          * Never raises.
        """
        try:
            self._adapter.ensure_table()
            table = self._adapter._table
            if table is None or "data_year" not in table.schema.names:
                # No data_year column to probe — assume present.
                return True
            current_count = int(table.count_rows())
            cache = getattr(self, "_distinct_years_cache", None)
            cache_count = getattr(self, "_distinct_years_count", None)
            if cache is not None and cache_count == current_count:
                # Cache hit — row-count matches.
                return int(year) in cache
            # Cache miss or stale — refresh via the cheap probe.
            distinct = self._probe_distinct_years()
            self._distinct_years_cache = distinct
            self._distinct_years_count = current_count
            return int(year) in distinct
        except Exception as exc:
            # Documented conservative default: assume data is present.
            logger.warning(
                f"has_current_year_data probe failed; assuming present: {exc}"
            )
            return True

    def _probe_distinct_years(self) -> set:
        """Cheap SELECT-DISTINCT-year probe against the LanceDB table.

        Default implementation uses a single-column `to_pandas` projection
        (NOT a full row fetch). The result set is bounded by the number of
        distinct admission years, which is tiny (1–N over the lifetime of
        the system), so the projection cost is negligible.

        Tests may monkey-patch this method to avoid touching LanceDB.
        """
        import pandas as pd  # local import to keep module-import light
        # Same projection-pushdown reason as _detect_current_admission_year.
        # Verified on 0.29.2: no default row cap — a 2138-row table returns 2138,
        # so the DISTINCT is computed over every row, not a truncated head.
        df = self._adapter._table.search().select(["data_year"]).to_pandas()
        distinct: set = set()
        for value in df["data_year"].tolist():
            if value is None:
                continue
            # pandas may return NaN for missing values; pd.isna covers that.
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            try:
                distinct.add(int(value))
            except (ValueError, TypeError):
                # Non-integer junk in data_year column — skip, do not crash.
                continue
        return distinct

    def get_admission_context(self) -> Dict[str, Any]:
        """Single helper consumed by S14.3 / S14.4.

        Composes (per call — no import-freeze):
          * `year`     : current admission year from `temporal_authority`
          * `has_data` : whether the index has any chunk for that year
          * `season`   : `SeasonPhase` from the pure season-state machine

        Returns:
            `{"year": int, "has_data": bool, "season": SeasonPhase}`
        """
        from services.temporal_authority import get_current_admission_year
        from services.season import get_season_phase  # S14.1 pure machine
        year = int(get_current_admission_year())
        has_data = self.has_current_year_data(year)
        season = get_season_phase(year, has_data)
        return {"year": year, "has_data": has_data, "season": season}

    def apply_post_fetch_filters(
        self,
        rows: List[Dict[str, Any]],
        metadata_filter: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """The post-fetch eligibility rule every retrieval arm must obey.

        Extracted from `retrieve()` so the dense arm and the BM25 arm
        (`HybridSearchService`) apply ONE rule rather than two copies that
        drift — a missing second copy is exactly how the BM25 arm ended up
        with no filter at all while dense dropped prior-year rows.
        """
        if not rows:
            return rows

        temporal_intent = None
        admission_ctx: Optional[Dict[str, Any]] = None
        classification_query = (
            query if query is not None else (metadata_filter or {}).get("query")
        )
        if classification_query:
            try:
                from services.temporal_intent import (
                    TemporalIntent,
                    classify_temporal,
                )
                admission_ctx = self.get_admission_context()
                year = int(admission_ctx["year"])
                has_data = bool(admission_ctx["has_data"])
                temporal_intent = classify_temporal(classification_query, year)
                admission_ctx = {
                    "year": year,
                    "has_data": has_data,
                    "season": admission_ctx["season"],
                }
            except Exception as exc:
                # Fail-safe: a year-resolution error (e.g. APP_ENV=prod with
                # CURRENT_ADMISSION_YEAR unset → RuntimeError) must NOT crash
                # retrieval. Fall back to no hard filter (legacy soft path).
                logger.warning(f"Temporal classification skipped: {exc}")
                temporal_intent = None
                admission_ctx = None

        rows = self._apply_filter(rows, metadata_filter)
        return self._apply_temporal_filter(
            rows,
            metadata_filter,
            temporal_intent=temporal_intent,
            admission_ctx=admission_ctx,
        )

    def _apply_filter(
        self,
        rows: List[Dict[str, Any]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not metadata_filter:
            return rows

        # `query` is a HINT, not a constraint: it exists so the temporal
        # classifier can read the user's wording, and no row carries a
        # `query` field. Matching it as equality dropped EVERY row, which
        # made the dense arm return EMPTY_RESULT on every hybrid call
        # (HybridSearchService injects this key). Same reasoning as
        # `or_conditions`, which is likewise skipped here.
        def match_condition(row: Dict[str, Any], cond: Dict[str, Any]) -> bool:
            raw = row.get("metadata_json") or row.get("metadata") or "{}"
            metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
            for key, value in cond.items():
                if key in ("or_conditions", "query"):
                    continue
                row_value = row.get(key)
                if row_value is None:
                    row_value = metadata.get(key)
                if row_value != value:
                    return False
            return True

        or_conditions = metadata_filter.get("or_conditions")
        if or_conditions:
            return [row for row in rows if any(match_condition(row, c) for c in or_conditions)]

        return [row for row in rows if match_condition(row, metadata_filter)]

    def _normalize_rows(self, rows: List[Dict[str, Any]]) -> List[RetrievedDocument]:
        documents: List[RetrievedDocument] = []
        for row in rows:
            text = row.get("text")
            if not text or not isinstance(text, str):
                continue

            raw = row.get("metadata_json") or row.get("metadata") or "{}"
            metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
            distance = row.get("_distance")
            score = float(row.get("score", 0.0))
            if distance is not None:
                score = 1.0 / (1.0 + float(distance))

            # M3-N4b: propagate degraded_filter flag into metadata
            if row.get("_degraded_filter"):
                metadata["degraded_filter"] = True

            documents.append(
                RetrievedDocument(
                    text=text,
                    source=row.get("source") or get_legacy_source({"metadata": metadata}) or None,
                    chunk_id=str(row.get("chunk_id") or row.get("id") or ""),
                    metadata=metadata,
                    score=score,
                    point_id=str(row.get("id")) if row.get("id") is not None else None,
                )
            )
        return documents

    def fetch_by_id(self, chunk_id: str) -> Optional[RetrievedDocument]:
        """Fetch a single chunk by its id using server-side filter (no full-table scan).

        Used by the Phase E aggregation booster to inject canonical summary chunks.

        Ticket 45 (Loi B) — two ID spaces, resolved in order:
          1. ``id`` column (hash / physical point id). This is the original
             contract and the path GraphRAG's single-fetch uses; unchanged.
          2. ``doc_id`` column (logical/symbolic id, e.g.
             ``tuyensinh_overview_2026_v2``). Booster rules reference symbolic
             ids that live ONLY in ``doc_id`` — the id lookup could never
             resolve them, so the booster silently de-boosted
             (``BOOSTER_CHUNK_MISS``). When the doc_id path matches several
             physical rows, ``_pick_promoted_row`` selects the promoted one
             deterministically.

        ``chunk_id`` on the returned document is the ID the caller asked for
        whenever it was resolved via ``doc_id`` (symbolic in == symbolic out,
        so the booster's seen_ids/retag share the requested space); on the
        hash path it keeps the row's own id semantics. ``fetch_by_ids`` (the
        batch/hash path) is intentionally unchanged.
        """
        try:
            self._adapter.ensure_table()
            # Use server-side 'id IN (...)' predicate to avoid full to_pandas() scan.
            rows = self._adapter.fetch_rows_by_ids([chunk_id])
            via = "id"
            if not rows:
                # Ticket 45: symbolic ids live in doc_id, not the hash id column.
                rows = self._adapter.fetch_rows_by_doc_ids([chunk_id])
                via = "doc_id"
            if not rows:
                return None
            row = _pick_promoted_row(rows) if len(rows) > 1 else rows[0]
            # T15 belt: server-side predicate already excludes tombstoned
            # rows; this guard catches any caller path that bypasses it.
            _ts = row.get("is_tombstone")
            if _ts is None:
                _ts = (json.loads(row.get('metadata_json') or '{}') or {}).get('is_tombstone')
            if _ts:
                return None
            raw_meta = row.get('metadata_json')
            metadata = json.loads(raw_meta) if isinstance(raw_meta, str) and raw_meta else {}
            # T15 effectiveness law: never serve a non-current / out-of-window
            # row through the booster or GraphRAG single-fetch. NULL = KEEP.
            if _drop_by_effectiveness(row, metadata, _current_as_of_int()):
                return None
            return RetrievedDocument(
                text=row.get('text', '') if 'text' in row else '',
                source=row.get('source') or get_legacy_source({'metadata': metadata}),
                chunk_id=chunk_id if via == "doc_id" else (row.get('chunk_id') or row.get('id') or ""),
                metadata=metadata,
                score=1.0,  # injected priority
                point_id=str(row.get('id')) if row.get('id') is not None else None,
            )
        except Exception as e:
            logger.warning(f"fetch_by_id({chunk_id}) failed: {e}")
            return None

    def fetch_by_ids(self, chunk_ids: List[str]) -> Dict[str, RetrievedDocument]:
        """Batch-fetch many chunks by their chunk_ids in a SINGLE filtered query.

        Performance: replaces the previous `to_pandas()` full-table scan
        with a server-side `id IN (...)` predicate pushed down to LanceDB
        (`LanceDBAdapter.fetch_rows_by_ids`). Only matching rows are
        returned; no full materialization. The return value is a
        `{chunk_id: RetrievedDocument}` map; ids not found are simply
        absent from the dict (caller treats the miss as `None`).

        Mirrors `fetch_by_id` construction exactly (same `score=1.0`
        injected-priority sentinel, same metadata_json parse) so the
        GraphRAG expander's downstream `score=0.0` neutralization stays
        the only behavior change. Never raises into the retrieve path.
        """
        if not chunk_ids:
            return {}
        try:
            rows = self._adapter.fetch_rows_by_ids(chunk_ids)
            if not rows:
                return {}

            result: Dict[str, RetrievedDocument] = {}
            for row in rows:
                row_id = row.get('id')
                if row_id is None or str(row_id) in result:
                    continue
                raw_meta = row.get('metadata_json')
                metadata = json.loads(raw_meta) if isinstance(raw_meta, str) and raw_meta else {}
                # T15 belt: skip tombstoned rows (server-side predicate
                # already excludes them; this catches bypassed callers).
                _ts = row.get('is_tombstone')
                if _ts is None:
                    _ts = metadata.get('is_tombstone')
                if _ts:
                    continue
                # T15 effectiveness law: same as fetch_by_id, for the
                # GraphRAG batched expander. NULL/absent = KEEP.
                if _drop_by_effectiveness(row, metadata, _current_as_of_int()):
                    continue
                result[str(row_id)] = RetrievedDocument(
                    text=row.get('text', ''),
                    source=row.get('source') or get_legacy_source({'metadata': metadata}),
                    chunk_id=row.get('chunk_id') or str(row_id),
                    metadata=metadata,
                    score=1.0,  # injected priority — same sentinel as fetch_by_id
                    point_id=str(row_id),
                )
            return result
        except Exception as e:
            logger.warning(f"fetch_by_ids({len(chunk_ids)} ids) failed: {e}")
            return {}

    def _calculate_confidence(self, documents: List[RetrievedDocument]) -> float:
        top_scores = [doc.score for doc in documents[:3]]
        if len(top_scores) == 1:
            return top_scores[0]
        if len(top_scores) == 2:
            return 0.7 * top_scores[0] + 0.3 * top_scores[1]
        return 0.5 * top_scores[0] + 0.3 * top_scores[1] + 0.2 * top_scores[2]

    def check_collection(self) -> Dict[str, Any]:
        exists = self._adapter.table_exists()
        if not exists:
            return {"exists": False, "error": f"Table '{self._config.table_name}' not found"}
        return {
            "exists": True,
            "name": self._config.table_name,
            "vectors_count": self._adapter.count(),
            "vector_dim": self._adapter.vector_dim(),
            "status": "ready",
        }

    async def healthcheck(self) -> bool:
        """Can this retriever serve? Used by /admin/reload-table before swapping.

        The route calls this on the *candidate* retriever, so a table that does
        not exist, or exists but is empty, must fail here rather than being
        swapped in and failing on the first query. Probes the real table through
        :meth:`check_collection` instead of returning a constant: a green
        healthcheck that cannot go red is not a healthcheck.

        ``to_thread`` because the probe hits LanceDB synchronously; the route
        awaits this on the event loop and must not block other requests.
        """
        try:
            status = await asyncio.to_thread(self.check_collection)
        except Exception as exc:
            logger.warning(
                f"healthcheck: table {self._config.table_name!r} probe raised {exc!r}"
            )
            return False
        # An empty table is a real failure mode for a swap: it would serve zero
        # documents for every query, which looks like "no data for your
        # question" rather than an outage.
        return bool(status.get("exists")) and int(status.get("vectors_count") or 0) > 0

    async def close(self) -> None:
        """Release the LanceDB handle held by this retriever.

        Exists so /admin/reload-table's ``_safe_close`` has something to call on
        the *old* retriever after a swap. Dropping the adapter reference is the
        whole job — LanceDB connections are not pooled here — but it must be
        idempotent: the route also closes a retriever it built and then failed to
        healthcheck.
        """
        adapter = getattr(self, "_adapter", None)
        self._adapter = None
        table = getattr(adapter, "_table", None)
        if table is not None:
            adapter._table = None


_default_retriever: Optional[LanceDBRetriever] = None


def get_retriever() -> LanceDBRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = LanceDBRetriever.from_env()
    return _default_retriever
