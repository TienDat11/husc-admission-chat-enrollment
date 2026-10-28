"""
HybridSearchService — dense + BM25 sparse retrieval with RRF fusion.

Works natively with RetrievedDocument (no adapter needed).
Integrates directly with LanceDBRetriever.

The sparse half is analyzed by ``services.vietnamese_analyzer`` (diacritic
folding + punctuation-aware tokens + exact-code postings), applied identically
to the indexed chunks and to the query. Dense retrieval and RRF (k=60) are
untouched by that: the analyzer only decides which chunks the sparse list
offers to the fusion.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from services.lancedb_retrieval import (
    _current_as_of_int,
    _drop_by_effectiveness,
    CURRENT_ADMISSION_YEAR,
    LanceDBRetriever,
    RetrievedDocument,
    RetrievalResult,
    YEAR_BOOST_FACTOR,
    OLD_YEAR_DEMOTE_FACTOR,
)
from services.vietnamese_analyzer import (
    ANALYZER_VERSION,
    analyze,
)
from config.settings import RAGSettings

logger = logging.getLogger(__name__)

# Exact-code boost (ticket 07).
#
# A code query must not be diluted by tokenization: "Thông tư 08/2022/TT-BGDĐT"
# tokenizes to 08/2022/tt/bgddt, where 08 and 2022 are corpus-wide noise, and
# "tổ hợp A00, A01" used to keep the comma attached to the code, so the token
# matched nothing at all. So chunks that DECLARE a code the query names get an
# additive boost on their BM25 score.
#
# Magnitude, measured on the served 1679-chunk HUSC corpus (max raw BM25 per
# query, no boost): "A00" 4.81, "7480201" 6.47, "học phí" 7.33, "điểm chuẩn
# 2026" 10.38, "xét tuyển tổ hợp A00 A01 D01" 19.21, "thông tư
# 08/2022/TT-BGDĐT quy chế tuyển sinh" 35.17. +5.0 per matched code is worth
# about one strong term: enough to put a code-bearing chunk at the head of a
# short code query, deliberately NOT enough to override a long query whose
# terms already point elsewhere.
EXACT_CODE_BOOST = 5.0
# A chunk naming several of the query's codes is a better match, but the boost
# must not scale without bound on a "list every tổ hợp" query.
EXACT_CODE_MAX_MATCHES = 3


class HybridSearchService:
    """
    Hybrid dense + BM25 retrieval with RRF fusion.

    Lifecycle:
      1. __init__(lancedb_retriever, settings)
      2. build_bm25_index() -> bool   (call once at startup)
      3. await retrieve(query, query_vector, top_k) -> RetrievalResult
    """

    def __init__(self, lancedb_retriever: LanceDBRetriever, settings: RAGSettings) -> None:
        self._retriever = lancedb_retriever
        self._settings = settings
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_docs: List[RetrievedDocument] = []
        # T15 effectiveness law: flat eligibility bounds per chunk_id,
        # so the per-request sparse filter can drop a chunk that expired
        # AFTER the index was built (no rebuild needed). Same key the RRF
        # fusion uses; absent key = un-stamped = KEEP.
        self._corpus_eligibility: Dict[str, Dict[str, Any]] = {}
        # Folded exact codes -> positions in `_corpus_docs`. Built at index time
        # with the same detector the query side uses, so a code is matched by
        # string equality on canonical form (never by substring).
        self._code_postings: Dict[str, List[int]] = {}
        # Identity of the indexed corpus — see `corpus_version`.
        self._corpus_version: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Index Build
    # ------------------------------------------------------------------

    def build_bm25_index(self) -> bool:
        """
        Scan LanceDB, build in-memory BM25 index.

        Uses column-selective to_pandas() to avoid loading 4096-dim vectors.
        Excludes superseded chunks from BM25 corpus (temporal freshness).
        Returns True on success, False on failure (caller should set service = None).
        """
        try:
            adapter = self._retriever._adapter
            table = adapter._table

            cols = ["text", "source", "chunk_id", "metadata_json", "id", "release_id", "is_tombstone",
                    "status", "effective_from", "effective_to"]
            available_cols = table.schema.names
            load_cols = [c for c in cols if c in available_cols]

            # LanceDB API tương thích nhiều version: dùng select() khi có
            # (avoid loading the 1024-dim vector column), fallback to_pandas()
            # rồi project columns ở pandas-level. Cả hai path đều an toàn.
            try:
                if hasattr(table, "select") and callable(getattr(table, "select")):
                    df = table.select(load_cols).to_pandas()
                else:
                    df = table.to_pandas()
                    df = df[[c for c in load_cols if c in df.columns]]
            except TypeError:
                # Phòng trường hợp .select() có signature khác — fallback an toàn
                df = table.to_pandas()
                df = df[[c for c in load_cols if c in df.columns]]

            if df.empty:
                logger.warning("HybridSearchService: LanceDB table is empty, BM25 not built")
                return False

            tokenized_corpus: List[List[str]] = []
            self._corpus_docs = []
            self._corpus_eligibility = {}
            self._code_postings = {}
            releases: set = set()
            meta_releases: set = set()
            digest = hashlib.sha256()

            for _, row in df.iterrows():
                # Exclude superseded chunks from BM25 index
                raw_meta = row.get("metadata_json", "{}")
                meta = json.loads(raw_meta) if isinstance(raw_meta, str) else (raw_meta or {})
                if meta.get("is_superseded", False):
                    continue
                # T15: a tombstoned row is never in the BM25 corpus.
                # The top-level column is the primary source; the metadata
                # mirror is a fallback for tables where only one is stamped.
                #
                # NULL trap (pandas): `is_tombstone` is nullable; a missing
                # stamp arrives as `float('nan')` inside a DataFrame, and
                # `nan is None` is False while `bool(nan)` is True — so a
                # bare truthiness check would silently drop every un-stamped
                # chunk. Treat NaN and None as "not tombstoned".
                is_tombstone = row.get("is_tombstone")
                if isinstance(is_tombstone, float) and math.isnan(is_tombstone):
                    is_tombstone = None
                if is_tombstone is None:
                    is_tombstone = meta.get("is_tombstone", False)
                if is_tombstone is not None and is_tombstone:
                    continue
                # T15 effectiveness law: flat row + NaN-aware window,
                # same rule as the dense belt in lancedb_retrieval.
                if _drop_by_effectiveness(dict(row), meta, _current_as_of_int()):
                    continue

                text = str(row.get("text", ""))
                chunk_id = str(row.get("chunk_id") or row.get("id") or "")

                # Index side of the analyzer. The chunk's own text is the only
                # token source: `sparse_terms`, where a table carries it, was
                # produced by a different (chunker-side) tokenizer, and mixing
                # two analyzers is what makes a sparse index unqueryable.
                analyzed = analyze(text)
                tokens = analyzed.tokens or [""]

                position = len(self._corpus_docs)
                for code in analyzed.codes:
                    self._code_postings.setdefault(code, []).append(position)

                tokenized_corpus.append(tokens)
                self._corpus_docs.append(
                    RetrievedDocument(
                        text=text,
                        source=str(row.get("source") or meta.get("source") or ""),
                        chunk_id=chunk_id,
                        metadata=meta,
                        score=0.0,
                        point_id=None,
                    )
                )
                # T15 effectiveness law: snapshot flat bounds (NaN-coerced)
                # so per-request filter applies SAME clock dense arm uses —
                # chunk expiring after build drops on next retrieve().
                _ef = row.get("effective_from")
                if isinstance(_ef, float) and math.isnan(_ef):
                    _ef = None
                _et = row.get("effective_to")
                if isinstance(_et, float) and math.isnan(_et):
                    _et = None
                _st = row.get("status")
                if isinstance(_st, float) and math.isnan(_st):
                    _st = None
                self._corpus_eligibility[chunk_id] = {
                    "status": _st,
                    "effective_from": _ef,
                    "effective_to": _et,
                    "is_tombstone": is_tombstone,
                }

                digest.update(chunk_id.encode("utf-8"))
                digest.update(b"\x00")
                digest.update(text.encode("utf-8"))
                digest.update(b"\n")

                # Two places a release id can live, recorded separately rather
                # than silently preferring one:
                #   * the top-level `release_id` column — what the ingest stamps
                #     on the row as a whole;
                #   * `metadata_json.release_id` — what the chunk itself carries,
                #     spread from the chunker's own record.
                # They are the same value once the publish path is consistent;
                # a disagreement means the row stamp is out of sync with the
                # chunk and is reported below, not papered over.
                release = row.get("release_id") if "release_id" in load_cols else None
                if isinstance(release, str) and release:
                    releases.add(release)
                meta_release = meta.get("release_id")
                if isinstance(meta_release, str) and meta_release:
                    meta_releases.add(meta_release)

            if not tokenized_corpus:
                logger.warning(
                    "HybridSearchService: no indexable chunks (all superseded?), BM25 not built"
                )
                return False

            self._bm25 = BM25Okapi(tokenized_corpus)
            if len(releases) > 1:
                # A served table spanning two releases is a partial publish: the
                # sparse index would then be built from a mix the dense side
                # does not describe either. Loud, not fatal — the dense side
                # serves the same mixed table, so failing here would only take
                # hybrid search down without fixing the corpus.
                logger.warning(
                    "HybridSearchService: BM25 corpus spans %d release_ids %s — "
                    "the serving table is a partial publish",
                    len(releases),
                    sorted(releases),
                )
            if meta_releases and releases != meta_releases:
                # The chunks say one thing and the row stamp says another: the
                # ingest column is out of sync with the chunks it wrote. Named
                # loudly because every downstream consumer keying staleness on
                # the column would otherwise trust the wrong value.
                logger.error(
                    "HybridSearchService: release_id column %s disagrees with "
                    "metadata_json.release_id %s — the serving table's row stamp "
                    "is out of sync with the chunks",
                    sorted(releases),
                    sorted(meta_releases),
                )
            self._corpus_version = {
                "analyzer": ANALYZER_VERSION,
                "release_ids": sorted(releases),
                "metadata_release_ids": sorted(meta_releases),
                "chunk_count": len(self._corpus_docs),
                "corpus_sha256": digest.hexdigest(),
                "table": getattr(getattr(self._retriever, "_config", None), "table_name", None),
                "code_count": len(self._code_postings),
            }
            logger.info(
                f"HybridSearchService: BM25 index built — {len(self._corpus_docs)} chunks, "
                f"analyzer={ANALYZER_VERSION}, release_ids={sorted(releases) or 'unset'}, "
                f"metadata_release_ids={sorted(meta_releases) or 'unset'}, "
                f"corpus_sha256={digest.hexdigest()[:12]}, codes={len(self._code_postings)}"
            )
            return True

        except Exception as e:
            logger.error(f"HybridSearchService: BM25 build failed: {e}", exc_info=True)
            return False

    @property
    def corpus_version(self) -> Dict[str, Any]:
        """Identity of the corpus the BM25 index was built from.

        ``release_ids`` comes from the table's own ``release_id`` column, so the
        sparse index cannot silently drift from the index being served: a
        re-publish that changes the release shows up here (and in
        ``corpus_sha256``, which covers every indexed chunk id + text).

        ``metadata_release_ids`` is the release id the indexed chunks carry in
        their own metadata. Both are recorded because the two can disagree (a
        row stamp out of sync with the chunks it wrote) — in that case an error
        is logged naming both rather than one silently winning.

        Empty dict until :meth:`build_bm25_index` succeeds.
        """
        return dict(self._corpus_version)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> RetrievalResult:
        """
        Hybrid retrieval: dense search + BM25 sparse -> RRF fusion.

        Falls back to dense-only if BM25 index unavailable.
        NOTE: retrieve() is async but calls sync LanceDB — no run_in_threadpool needed
        because LanceDB is embedded (in-process); there is no network or kernel I/O
        that would block the event loop. If LanceDB is ever switched to remote mode, revisit.
        """
        # Inject query text into metadata_filter so temporal filter can detect
        # historical intent (e.g. "so sánh 2024 và 2025")
        mf = metadata_filter or {}
        mf = {**mf, "query": query}

        dense_result: RetrievalResult = self._retriever.retrieve(
            query_vector=query_vector,
            top_k=top_k * 2,
            metadata_filter=mf,
        )
        dense_docs: List[RetrievedDocument] = dense_result.documents

        if self._bm25 is None:
            logger.warning("HybridSearchService: BM25 not ready, using dense-only")
            return dense_result

        sparse_docs = self._bm25_search(query, top_k=top_k * 2)
        sparse_docs = self._apply_post_fetch_filter(sparse_docs, mf)
        fused = self._rrf_fusion(dense_docs, sparse_docs, top_k=top_k)
        confidence = sum(d.score for d in fused) / len(fused) if fused else 0.0
        return RetrievalResult(documents=fused, confidence=confidence)

    def _apply_post_fetch_filter(
        self,
        docs: List[RetrievedDocument],
        metadata_filter: Dict[str, Any],
    ) -> List[RetrievedDocument]:
        """Put BM25 candidates through the SAME post-fetch rule as dense.

        The BM25 index is an in-memory scan of the whole table, so without
        this the sparse arm would fuse in chunks that the dense arm had
        excluded (superseded, prior-year) — the filter asymmetry ticket 09
        criterion 4 forbids. Rows are shaped the way the retriever's filter
        expects (`metadata_json` may be a dict) and survivors are matched
        back by position.
        """
        if not docs:
            return docs

        rows = [
            {
                "text": doc.text,
                "source": doc.source,
                "chunk_id": doc.chunk_id,
                "metadata_json": doc.metadata or {},
            }
            for doc in docs
        ]
        # T15 effectiveness law: re-attach the flat bounds
        # snapshotted at index build so a chunk that expired
        # (or flipped status) after the build is dropped HERE —
        # the dense arm reads the flat columns directly, the
        # sparse arm must not go stale. Absent key = un-stamped.
        for row in rows:
            eligibility = self._corpus_eligibility.get(row["chunk_id"])
            if eligibility:
                row.update(eligibility)
        kept_ids = {
            row["chunk_id"]
            for row in self._retriever.apply_post_fetch_filters(rows, metadata_filter)
        }
        return [doc for doc in docs if doc.chunk_id in kept_ids]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedDocument]:
        """Return top-k docs from BM25 index as RetrievedDocument list."""
        analyzed = analyze(query)
        tokens = analyzed.tokens
        if not tokens:
            logger.warning("_bm25_search called with empty token list — returning []")
            return []

        scores = self._bm25.get_scores(tokens)
        if analyzed.codes:
            scores = self._apply_exact_code_boost(scores, analyzed.codes)

        scored = sorted(zip(scores, self._corpus_docs), key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scored[:top_k * 3]:
            if score <= 0:
                break
            # Temporal boost/demote: adjust BM25 score by data_year
            adjusted_score = self._apply_temporal_bm25_boost(doc, score)
            results.append(
                RetrievedDocument(
                    text=doc.text,
                    source=doc.source,
                    chunk_id=doc.chunk_id,
                    metadata=doc.metadata,
                    score=adjusted_score,
                    point_id=doc.point_id,
                )
            )

        # Re-sort by adjusted score and return top_k
        results.sort(key=lambda d: d.score, reverse=True)
        return results[:top_k]

    def _apply_exact_code_boost(
        self, scores: Sequence[float], codes: Tuple[str, ...]
    ) -> List[float]:
        """Add :data:`EXACT_CODE_BOOST` to chunks that declare the query's codes.

        ``codes`` are already canonical (folded, whitespace-free) and the
        postings were built with the same detector, so this is exact string
        equality — a query for ``7480201`` does not boost ``7480201VJ``, and a
        code written ``A00,`` is still the same code as ``A00``.
        """
        matched: Dict[int, int] = {}
        for code in codes:
            for position in self._code_postings.get(code, ()):
                matched[position] = matched.get(position, 0) + 1

        boosted = list(scores)
        for position, hits in matched.items():
            boosted[position] += EXACT_CODE_BOOST * min(hits, EXACT_CODE_MAX_MATCHES)
        return boosted

    @staticmethod
    def _apply_temporal_bm25_boost(doc: RetrievedDocument, raw_score: float) -> float:
        """Apply temporal scoring to BM25 results: boost current year, demote old."""
        meta = doc.metadata or {}
        chunk_year = meta.get("data_year") or meta.get("year")
        if isinstance(chunk_year, str):
            try:
                chunk_year = int(chunk_year)
            except (ValueError, TypeError):
                chunk_year = None

        if chunk_year is None:
            return raw_score

        if chunk_year == CURRENT_ADMISSION_YEAR:
            return raw_score * YEAR_BOOST_FACTOR
        elif chunk_year < CURRENT_ADMISSION_YEAR:
            return raw_score * OLD_YEAR_DEMOTE_FACTOR
        return raw_score

    def _rrf_fusion(
        self,
        dense_docs: List[RetrievedDocument],
        sparse_docs: List[RetrievedDocument],
        top_k: int,
        k: int = 60,
    ) -> List[RetrievedDocument]:
        """
        Reciprocal Rank Fusion (RRF) over dense + sparse result lists.

        RRF score = sum(weight_i / (k + rank_i))
        k=60 is the standard parameter from Cormack et al. SIGIR 2009.
        """
        dense_weight = self._settings.HYBRID_FUSION_DENSE_WEIGHT
        sparse_weight = self._settings.HYBRID_FUSION_SPARSE_WEIGHT

        scores: Dict[str, float] = {}
        docs_map: Dict[str, RetrievedDocument] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + dense_weight / (k + rank)
            if cid not in docs_map:
                docs_map[cid] = doc

        for rank, doc in enumerate(sparse_docs, start=1):
            cid = doc.chunk_id
            scores[cid] = scores.get(cid, 0.0) + sparse_weight / (k + rank)
            if cid not in docs_map:
                docs_map[cid] = doc

        sorted_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        fused = []
        for cid in sorted_ids[:top_k]:
            doc = docs_map[cid]
            fused.append(
                RetrievedDocument(
                    text=doc.text,
                    source=doc.source,
                    chunk_id=doc.chunk_id,
                    metadata=doc.metadata,
                    score=scores[cid],
                    point_id=doc.point_id,
                )
            )

        return fused
