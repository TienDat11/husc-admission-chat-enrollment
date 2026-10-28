"""Aggregation Chunk Booster — inject high-value summary chunks into retrieval results.

When the query is enumeration / listing / counting / superlative / comparison,
prepend relevant aggregation chunks to baseline retrieval so the generator has a
summary view alongside per-ngành chunks.

Phase E — Slice E.2.

G1-T2 / G1-T3 (durability, S14.x): the current admission year is now derived
at module init from `temporal_authority.get_current_admission_year()`. The
year-suffixed chunk-ids (`_2026` style) and year-literal regex patterns are
built at runtime from this authority. Explicit historical rules (e.g. 2025)
are kept as intentional literal rules. The chunk-id set at year=2026 is
byte-identical to the golden fixture `tests/services/fixtures/
booster_chunkids_2026.golden.json` (45 ids).

A `BOOSTER_CHUNK_MISS` counter and a visible WARN log are emitted whenever
`fetch_by_id` returns None for an aggregation chunk — so silent de-boost
failures are observable in production.
"""
from __future__ import annotations

import re
from dataclasses import replace
from threading import Lock
from typing import List, Optional

from loguru import logger

from src.services.aggregation_rules import (
    _AGGREGATION_RULES as _AGGREGATION_RULES,
    _HISTORICAL_YEAR_1 as _HISTORICAL_YEAR_1,
    _YEAR as _YEAR,
    _YEAR_AGNOSTIC_SUFFIX as _YEAR_AGNOSTIC_SUFFIX,
    _chunk,
    _pat,
)
from src.services.lancedb_retrieval import RetrievedDocument
from src.services.log_redaction import safe_query


# ─── G1-T3: booster chunk-miss counter ──────────────────────────────────────
# Lightweight in-process counter (no Prometheus dep) so the WARN-vs-DEBUG
# regression in G1-T3 is unit-testable. Production metric export is wired
# elsewhere (see observability.request_tracker). Read with `.get()`.

class _BoosterChunkMiss:
    """Tiny thread-safe counter used by the booster to surface None-fetches."""

    def __init__(self) -> None:
        self._n = 0
        self._lock = Lock()

    def inc(self, n: int = 1) -> None:
        with self._lock:
            self._n += n

    def get(self) -> int:
        with self._lock:
            return self._n


BOOSTER_CHUNK_MISS = _BoosterChunkMiss()


def detect_aggregation_chunks(query: str) -> List[str]:
    """Return list of priority chunk_ids that should be injected for this query."""
    query_lower = query.lower()
    chunks_to_inject: List[str] = []
    seen = set()
    for rule in _AGGREGATION_RULES:
        for pattern in rule["patterns"]:
            if re.search(pattern, query_lower):
                for cid in rule["chunks"]:
                    if cid not in seen:
                        seen.add(cid)
                        chunks_to_inject.append(cid)
                break
    return chunks_to_inject


def boost_with_aggregation(
    query: str,
    baseline_docs: List[RetrievedDocument],
    lancedb_retriever,
    top_k: int = 5,
    max_inject: int = 2,
) -> List[RetrievedDocument]:
    """Inject aggregation chunks ahead of baseline if query matches.

    Args:
        query: original user query
        baseline_docs: baseline retrieved docs (post-rerank)
        lancedb_retriever: LanceDBRetriever instance to fetch by chunk_id
        top_k: total final docs (auto-expanded by `max_inject` when injection happens)
        max_inject: hard cap on injected aggregation chunks per query

    Returns:
        Final docs with aggregation chunks prepended (deduped)

    Observability:
        On every `fetch_by_id` that returns `None` WHILE INJECTION CAPACITY
        REMAINS (`len(injected) < max_inject`), the booster:
          - logs a WARNING with the chunk-id and query (G1-T3: visible, not debug)
          - increments the `BOOSTER_CHUNK_MISS` in-process counter
        Once the cap is full the ref could not have been injected anyway, so
        its absence is not a de-boost and is not counted (same observable
        count as the old fetch-then-break loop). Refs after the cap is full
        are still FETCHED, because a ref whose chunk already sits in the
        baseline needs the non-consuming `booster_injected` re-tag regardless
        of the cap (fusion-protection invariant).
    """
    inject_ids = detect_aggregation_chunks(query)
    if not inject_ids:
        return baseline_docs[:top_k]

    seen_ids = {d.chunk_id for d in baseline_docs if d.chunk_id}
    # Ticket 45 (Loi B): baseline docs come from vector search, where
    # _normalize_rows sets chunk_id = row id (a hash) and point_id = the
    # same hash. Booster refs are SYMBOLIC ids that resolve (via the
    # doc_id lookup) to a row whose id is that hash. Without a point_id
    # space, `cid in seen_ids` never matches, the fetched doc is injected
    # as a SECOND copy of a chunk already in the baseline, and the
    # baseline copy never gets its booster_injected protection tag.
    seen_points = {d.point_id for d in baseline_docs if d.point_id}
    # Track baseline docs that were already aggregate-relevant: re-tag them
    # with booster_injected=True so the GraphRAG fusion exempt-block keeps
    # them. This is a protection pass, not a new injection — it does NOT
    # consume a max_inject slot. Live bug: "HUSC đào tạo những ngành nào"
    # used to drop the full 28-major list (already in baseline) in favor
    # of the generic tuyensinh_overview.
    retag: dict[str, "RetrievedDocument"] = {}
    injected_docs: List[RetrievedDocument] = []
    for cid in inject_ids:
        if cid in seen_ids:
            existing = next((d for d in baseline_docs if d.chunk_id == cid), None)
            if existing is not None and not (existing.metadata or {}).get("booster_injected"):
                retag[cid] = replace(
                    existing,
                    metadata={**(existing.metadata or {}), "booster_injected": True},
                )
            continue
        try:
            doc = lancedb_retriever.fetch_by_id(cid)
        except Exception as e:
            # G1-T3: visible failure, but cap-gated — with the cap full this
            # ref could not have been injected, so it is not a de-boost.
            if len(injected_docs) < max_inject:
                BOOSTER_CHUNK_MISS.inc()
                logger.warning(
                    f"booster_chunk_miss: fetch_by_id({cid!r}) raised "
                    f"{type(e).__name__}: {e} for query={safe_query(query)}"
                )
            continue
        if not doc:
            # G1-T3: visible failure. Year-rollover (e.g. 2026→2027)
            # produces a None here; the previous `logger.debug` was
            # invisible to operators. Emit WARN + bump counter — but only
            # while an inject slot remains: with the cap full this ref
            # could not have been injected anyway, so counting it would be
            # a false alarm (same observable count as the old
            # fetch-then-break loop).
            if len(injected_docs) < max_inject:
                BOOSTER_CHUNK_MISS.inc()
                logger.warning(
                    f"booster_chunk_miss: fetch_by_id({cid!r}) returned None "
                    f"for query={safe_query(query)} (likely year-mismatch: chunk-id "
                    f"from year={_YEAR} but data not yet ingested)"
                )
            continue
        pid = getattr(doc, "point_id", None)
        if pid and pid in seen_points:
            # Point-space re-tag: the physical chunk is ALREADY in
            # the baseline (as a hash-keyed vector-search doc). Tag
            # that copy in place — keeping its rerank score, text
            # and source — and normalize its chunk_id to the
            # symbolic id the rule asked for, so seen_ids and
            # inject_ids share one space. Do NOT inject a duplicate
            # and do NOT consume a max_inject slot. Runs regardless
            # of the cap: fusion protection is not an injection.
            existing = next(
                (d for d in baseline_docs if d.point_id == pid), None
            )
            if existing is not None and existing.chunk_id and not (existing.metadata or {}).get(
                "booster_injected"
            ):
                retag[existing.chunk_id] = replace(
                    existing,
                    chunk_id=cid,
                    metadata={**(existing.metadata or {}), "booster_injected": True},
                )
            continue
        if len(injected_docs) >= max_inject:
            # Cap full: skip injecting THIS ref but keep scanning the
            # remaining refs — a later ref whose chunk already sits in the
            # baseline still needs its non-consuming retag above. (The old
            # `break` here silently dropped that protection.)
            continue
        # S15.5 / CF-2: tag injected docs so the precision-cutoff
        # (separate module, Lane A) can EXEMPT them from the
        # relative rerank-score drop. RetrievedDocument is a
        # frozen dataclass, so we reconstruct with merged
        # metadata and keep the score from `fetch_by_id`
        # (booster docs retain their injected priority by design;
        # only the GraphRAG expander (S15.1) zero-bases scores).
        tagged = replace(
            doc,
            metadata={**doc.metadata, "booster_injected": True},
        )
        injected_docs.append(tagged)
        seen_ids.add(cid)

    if not injected_docs:
        # Nothing NEW injected, but an aggregate chunk may already be in
        # baseline and need the booster_injected protection tag (re-tag pass).
        # Without this branch the early return would drop the retag and the
        # fusion top-k slice could still evict the full-list chunk. When no
        # retag either, return the untouched baseline (true no-op).
        if not retag:
            return baseline_docs[:top_k]
        retagged_baseline = [
            retag.get(d.chunk_id, d) for d in baseline_docs
        ]
        return retagged_baseline[:top_k]

    logger.info(
        f"Aggregation booster: injected {len(injected_docs)} chunks "
        f"({[d.chunk_id for d in injected_docs]})"
    )

    # Expand top_k to keep room for original baseline relevance
    effective_top_k = top_k + len(injected_docs)
    injected_cids = {x.chunk_id for x in injected_docs}
    final: List[RetrievedDocument] = list(injected_docs)
    for d in baseline_docs:
        if d.chunk_id in injected_cids:
            continue
        if d.chunk_id in retag:
            final.append(retag[d.chunk_id])
        else:
            final.append(d)
    return final[:effective_top_k]
