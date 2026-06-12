"""
Retrieval Layer - LanceDB Interface

Fail-safe dense retrieval for embedded LanceDB backend.
Includes temporal filtering: auto-prefer newest year, exclude superseded.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger

from config.settings import RAGSettings
from src.infrastructure.lancedb_adapter import LanceDBAdapter

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
                import pandas as pd
                df = table.to_pandas(columns=["data_year"])
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
        # `adapter.search` is therefore always called with
        # `where_clause=None` — no code path may emit a `data_year` SQL
        # clause. (The param is retained for future non-year filters.)
        temporal_intent = None
        admission_ctx: Optional[Dict[str, Any]] = None
        classification_query = (
            query
            if query is not None
            else (metadata_filter or {}).get("query")
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

        try:
            rows = self._adapter.search(
                query_vector=query_vector,
                top_k=max(limit * 5, limit),
                where_clause=None,
            )
            rows = self._apply_filter(rows, metadata_filter)
            rows = self._apply_temporal_filter(
                rows,
                metadata_filter,
                temporal_intent=temporal_intent,
                admission_ctx=admission_ctx,
            )
            docs = self._normalize_rows(rows[:limit])
            if not docs:
                return RetrievalResult(
                    documents=[],
                    error_type=RetrievalError.EMPTY_RESULT,
                    error_message="No matching documents found",
                    confidence=0.0,
                )
            return RetrievalResult(documents=docs, confidence=self._calculate_confidence(docs))
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

            # Rule 2: Exclude superseded chunks (unless historical query)
            is_superseded = meta.get("is_superseded", False)
            if is_superseded and not is_historical_query:
                # Hard-drop (or push far down). For the hard-EXCLUDE path
                # we drop outright; for the soft path we demote with +100.
                if hard_exclude_active:
                    continue
                row["_distance"] = row.get("_distance", 999.0) + 100.0  # Push far down
                row["_temporal_demoted"] = True
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
        df = self._adapter._table.to_pandas(columns=["data_year"])
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

    def _apply_filter(
        self,
        rows: List[Dict[str, Any]],
        metadata_filter: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not metadata_filter:
            return rows

        def match_condition(row: Dict[str, Any], cond: Dict[str, Any]) -> bool:
            raw = row.get("metadata_json") or row.get("metadata") or "{}"
            metadata = json.loads(raw) if isinstance(raw, str) else (raw or {})
            for key, value in cond.items():
                if key == "or_conditions":
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

            documents.append(
                RetrievedDocument(
                    text=text,
                    source=row.get("source"),
                    chunk_id=row.get("chunk_id"),
                    metadata=metadata,
                    score=score,
                    point_id=str(row.get("id")) if row.get("id") is not None else None,
                )
            )
        return documents

    def fetch_by_id(self, chunk_id: str) -> Optional[RetrievedDocument]:
        """Fetch a single chunk by its chunk_id (no vector search).

        Used by the Phase E aggregation booster to inject canonical summary chunks.
        """
        try:
            self._adapter.ensure_table()
            df = self._adapter._table.to_pandas()
            match = df[df['id'] == chunk_id]
            if match.empty:
                return None
            row = match.iloc[0]
            raw_meta = row.get('metadata_json') if 'metadata_json' in df.columns else None
            metadata = json.loads(raw_meta) if isinstance(raw_meta, str) and raw_meta else {}
            return RetrievedDocument(
                text=row.get('text', '') if 'text' in df.columns else '',
                source=row.get('source') if 'source' in df.columns else None,
                chunk_id=row.get('id') if 'id' in df.columns else chunk_id,
                metadata=metadata,
                score=1.0,  # injected priority
                point_id=str(row.get('id')) if 'id' in df.columns and row.get('id') is not None else None,
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
                result[str(row_id)] = RetrievedDocument(
                    text=row.get('text', ''),
                    source=row.get('source'),
                    chunk_id=row.get('chunk_id', row_id),
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
            "status": "ready",
        }


_default_retriever: Optional[LanceDBRetriever] = None


def get_retriever() -> LanceDBRetriever:
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = LanceDBRetriever.from_env()
    return _default_retriever
