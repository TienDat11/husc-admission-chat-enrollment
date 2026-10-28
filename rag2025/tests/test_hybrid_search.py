"""Unit tests for HybridSearchService."""
from __future__ import annotations

import asyncio
import json
import unittest
from unittest.mock import MagicMock

import pandas as pd

from services.hybrid_search import HybridSearchService
from services.lancedb_retrieval import RetrievedDocument, RetrievalResult


def _make_doc(
    chunk_id: str,
    text: str = "sample text",
    score: float = 1.0,
    metadata: dict | None = None,
) -> RetrievedDocument:
    return RetrievedDocument(
        text=text,
        source="src",
        chunk_id=chunk_id,
        metadata=metadata or {},
        score=score,
        point_id=None,
    )


def _make_settings(use_hybrid=True, dense_w=0.6, sparse_w=0.4):
    s = MagicMock()
    s.USE_HYBRID_RETRIEVAL = use_hybrid
    s.HYBRID_FUSION_DENSE_WEIGHT = dense_w
    s.HYBRID_FUSION_SPARSE_WEIGHT = sparse_w
    return s


def _make_retriever(docs=None, release_id="R-2026-08-01"):
    retriever = MagicMock()
    docs = docs or []
    retriever.retrieve.return_value = RetrievalResult(documents=docs, confidence=0.5)
    # The sparse arm routes BM25 candidates through the retriever's post-fetch
    # rule. This stub models a retriever with no temporal filter active, so
    # rows pass through untouched — a bare MagicMock returns an EMPTY iterable
    # and would silently drop every BM25 candidate, hiding the sparse arm from
    # these tests. The rule itself is covered by
    # tests/services/test_lancedb_temporal_wiring.py.
    retriever.apply_post_fetch_filters.side_effect = lambda rows, *a, **k: rows
    # Mock adapter._table for BM25 build. Column set mirrors the served table
    # (release_id is the publish stamp; there is no sparse_terms column).
    table = MagicMock()
    table.schema.names = ["text", "source", "chunk_id", "metadata_json", "release_id"]
    df = (
        pd.DataFrame(
            {
                "text": [d.text for d in docs],
                "source": [d.source for d in docs],
                "chunk_id": [d.chunk_id for d in docs],
                "metadata_json": [json.dumps(d.metadata) for d in docs],
                "release_id": [release_id for _ in docs],
            }
        )
        if docs
        else pd.DataFrame()
    )
    # MagicMock's select().to_pandas() returns another MagicMock by default.
    # Make it return the real DataFrame so hybrid_search sees non-empty table.
    select_result = MagicMock()
    select_result.to_pandas.return_value = df
    table.select.return_value = select_result
    table.to_pandas.return_value = df
    retriever._adapter._table = table
    return retriever


class TestBuildBm25Index(unittest.TestCase):
    def test_build_bm25_index_success(self):
        docs = [_make_doc("c1", "tuyển sinh đại học"), _make_doc("c2", "học phí 2024")]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        result = svc.build_bm25_index()
        self.assertTrue(result)
        self.assertIsNotNone(svc._bm25)
        self.assertEqual(len(svc._corpus_docs), 2)

    def test_build_bm25_index_empty_table(self):
        retriever = _make_retriever([])
        svc = HybridSearchService(retriever, _make_settings())
        result = svc.build_bm25_index()
        self.assertFalse(result)
        self.assertIsNone(svc._bm25)

    def test_build_bm25_index_failure_graceful(self):
        retriever = MagicMock()
        retriever._adapter._table.schema.names = ["text"]
        retriever._adapter._table.to_pandas.side_effect = RuntimeError("DB error")
        svc = HybridSearchService(retriever, _make_settings())
        result = svc.build_bm25_index()
        self.assertFalse(result)
        self.assertIsNone(svc._bm25)


class TestRetrieve(unittest.TestCase):
    def test_hybrid_retrieve_full_path(self):
        docs = [_make_doc("c1", "tuyển sinh đại học HUSC"), _make_doc("c2", "ngành học 2024")]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                svc.retrieve(query="tuyển sinh", query_vector=[0.1] * 10, top_k=2)
            )
        finally:
            loop.close()
        self.assertIsInstance(result, RetrievalResult)
        self.assertGreater(len(result.documents), 0)

    def test_hybrid_retrieve_bm25_not_ready_fallback(self):
        docs = [_make_doc("c1", "text")]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        # Do NOT build index — _bm25 is None
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                svc.retrieve(query="tuyển sinh", query_vector=[0.1] * 10, top_k=2)
            )
        finally:
            loop.close()
        # Should fall back to dense-only
        self.assertEqual(result.documents, docs)


class TestBm25Search(unittest.TestCase):
    def test_bm25_search_returns_relevant(self):
        docs = [
            _make_doc("c1", "tuyển sinh đại học"),
            _make_doc("c2", "học phí ngành công nghệ"),
            _make_doc("c3", "thời gian nộp hồ sơ"),
        ]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("học phí", top_k=3)
        # The doc containing "học phí" tokens should rank first
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0].chunk_id, "c2")

    def test_bm25_search_empty_query_guard(self):
        docs = [_make_doc("c1", "tuyển sinh")]
        retriever = _make_retriever(docs)
        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("", top_k=5)
        self.assertEqual(result, [])


class TestAnalyzerContract(unittest.TestCase):
    """The BM25 side must analyze index and query with the same analyzer."""

    def test_no_diacritics_query_finds_accented_chunk(self):
        docs = [
            _make_doc("c1", "Thời gian nộp hồ sơ xét tuyển năm 2026"),
            _make_doc("c2", "Học phí tín chỉ ngành Công nghệ thông tin là 600.000 đồng"),
            _make_doc("c3", "Điểm chuẩn các ngành năm 2025"),
        ]
        svc = HybridSearchService(_make_retriever(docs), _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("hoc phi tin chi", top_k=3)
        self.assertEqual(result[0].chunk_id, "c2")

    def test_accented_and_unaccented_queries_rank_identically(self):
        docs = [
            _make_doc("c1", "Điểm chuẩn ngành Sư phạm năm 2026"),
            _make_doc("c2", "Học phí ngành Công nghệ thông tin"),
            _make_doc("c3", "Thời gian nộp hồ sơ xét tuyển"),
            _make_doc("c4", "Chỉ tiêu tuyển sinh các ngành"),
        ]
        svc = HybridSearchService(_make_retriever(docs), _make_settings())
        svc.build_bm25_index()
        accented = [d.chunk_id for d in svc._bm25_search("học phí công nghệ", top_k=2)]
        plain = [d.chunk_id for d in svc._bm25_search("hoc phi cong nghe", top_k=2)]
        self.assertEqual(accented, plain)
        self.assertEqual(accented[0], "c2")

    def test_original_text_is_not_deformed_by_analysis(self):
        """The text served as source/citation must survive byte-identical."""
        original = "Ngưỡng đảm bảo chất lượng đầu vào năm 2026 (Điều 2, Thông tư 08/2022/TT-BGDĐT)"
        docs = [
            _make_doc("c1", original),
            _make_doc("c2", "Điểm chuẩn ngành Sư phạm năm 2026"),
            _make_doc("c3", "Thời gian nộp hồ sơ xét tuyển"),
        ]
        svc = HybridSearchService(_make_retriever(docs), _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("nguong dam bao chat luong", top_k=1)
        self.assertEqual(result[0].text, original)
        self.assertEqual(result[0].text.encode("utf-8"), original.encode("utf-8"))

    def test_chunk_text_overrides_any_table_supplied_sparse_terms(self):
        """One analyzer only: a table column tokenized elsewhere must not be
        used as the index side, or the query side can never agree with it."""
        docs = [
            _make_doc("c1", "Học phí ngành Công nghệ thông tin"),
            _make_doc("c2", "Điểm chuẩn ngành Sư phạm năm 2026"),
            _make_doc("c3", "Thời gian nộp hồ sơ xét tuyển"),
        ]
        retriever = _make_retriever(docs)
        table = retriever._adapter._table
        df = table.select.return_value.to_pandas.return_value
        df["sparse_terms"] = [["zzz", "qqq"] for _ in docs]
        table.schema.names = table.schema.names + ["sparse_terms"]
        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("học phí công nghệ", top_k=1)
        self.assertEqual(result[0].chunk_id, "c1")


class TestExactCodeBoost(unittest.TestCase):
    """A literal code query must not be diluted by tokenization."""

    def test_major_code_query_ranks_the_declaring_chunk_first(self):
        docs = [
            # Repeats the query's other terms but never names the code.
            _make_doc("noise", "Mã ngành mã ngành xét tuyển mã ngành xét tuyển mã ngành"),
            _make_doc("declares", "Ngành Công nghệ thông tin — Mã ngành: 7480201. Chỉ tiêu 400."),
        ]
        svc = HybridSearchService(_make_retriever(docs), _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("mã ngành 7480201", top_k=2)
        self.assertEqual(result[0].chunk_id, "declares")

    def test_to_hop_query_ranks_the_declaring_chunk_first(self):
        docs = [
            _make_doc("noise", "tổ hợp tổ hợp xét tuyển tổ hợp xét tuyển tổ hợp xét tuyển"),
            _make_doc("declares", "Toán, Vật lí, Hóa học — A00 — XTT"),
        ]
        svc = HybridSearchService(_make_retriever(docs), _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("xét tuyển tổ hợp A00", top_k=2)
        self.assertEqual(result[0].chunk_id, "declares")

    def test_document_number_query_ranks_the_declaring_chunk_first(self):
        docs = [
            _make_doc(
                "noise",
                "Thông tư năm 2022 quy chế tuyển sinh thông tư năm 2022 quy chế tuyển sinh",
            ),
            _make_doc("declares", "Thông tư số 08/2022/TT-BGDĐT, Điều 4.1"),
        ]
        svc = HybridSearchService(_make_retriever(docs), _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("Thông tư 08/2022/TT-BGDĐT", top_k=2)
        self.assertEqual(result[0].chunk_id, "declares")

    def test_comma_separated_code_list_matches(self):
        """``A00,`` used to be one token, so the code matched nothing."""
        docs = [
            _make_doc("c1", "Ngành Kỹ thuật phần mềm xét tổ hợp D01"),
            _make_doc("c2", "Ngành Công nghệ thông tin xét tổ hợp A00, D01"),
        ]
        svc = HybridSearchService(_make_retriever(docs), _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("tổ hợp A00", top_k=2)
        self.assertEqual(result[0].chunk_id, "c2")

    def test_code_absent_from_corpus_does_not_break_the_ranking(self):
        docs = [
            _make_doc("c1", "Học phí ngành Công nghệ thông tin"),
            _make_doc("c2", "Điểm chuẩn ngành Sư phạm năm 2026"),
            _make_doc("c3", "Thời gian nộp hồ sơ xét tuyển"),
            _make_doc("c4", "Chỉ tiêu tuyển sinh các ngành"),
        ]
        svc = HybridSearchService(_make_retriever(docs), _make_settings())
        svc.build_bm25_index()
        result = svc._bm25_search("học phí 9999999", top_k=1)
        self.assertEqual([d.chunk_id for d in result], ["c1"])


class TestCorpusVersion(unittest.TestCase):
    """The sparse index must be able to prove which corpus it was built from."""

    def test_records_release_id_analyzer_and_hash(self):
        from services.vietnamese_analyzer import ANALYZER_VERSION

        docs = [_make_doc("c1", "Học phí 2026"), _make_doc("c2", "Điểm chuẩn 2026")]
        svc = HybridSearchService(
            _make_retriever(docs, release_id="R-2026-08-01"), _make_settings()
        )
        svc.build_bm25_index()
        version = svc.corpus_version
        self.assertEqual(version["release_ids"], ["R-2026-08-01"])
        self.assertEqual(version["analyzer"], ANALYZER_VERSION)
        self.assertEqual(version["chunk_count"], 2)
        self.assertRegex(version["corpus_sha256"], r"^[0-9a-f]{64}$")

    def test_hash_changes_when_the_corpus_changes(self):
        """Identity must move when the indexed text does — otherwise drift is
        invisible."""
        first = HybridSearchService(
            _make_retriever([_make_doc("c1", "Học phí 2026")]), _make_settings()
        )
        first.build_bm25_index()
        second = HybridSearchService(
            _make_retriever([_make_doc("c1", "Học phí 2027")]), _make_settings()
        )
        second.build_bm25_index()
        self.assertNotEqual(
            first.corpus_version["corpus_sha256"], second.corpus_version["corpus_sha256"]
        )

    def test_empty_before_the_index_is_built(self):
        svc = HybridSearchService(_make_retriever(), _make_settings())
        self.assertEqual(svc.corpus_version, {})

    def test_absent_release_id_is_reported_as_unset_not_invented(self):
        retriever = _make_retriever([_make_doc("c1", "Học phí 2026")])
        retriever._adapter._table.schema.names = [
            "text",
            "source",
            "chunk_id",
            "metadata_json",
        ]
        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        self.assertEqual(svc.corpus_version["release_ids"], [])
        self.assertEqual(svc.corpus_version["metadata_release_ids"], [])

    def test_chunk_metadata_release_id_is_recorded_alongside_the_row_stamp(self):
        """A chunk's own release id is recorded even when the row stamp is
        absent — the chunk is the thing being served."""
        docs = [_make_doc("c1", "Học phí 2026", metadata={"release_id": "R-2026-09-01"})]
        retriever = _make_retriever(docs)
        retriever._adapter._table.schema.names = [
            "text",
            "source",
            "chunk_id",
            "metadata_json",
        ]
        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        self.assertEqual(svc.corpus_version["release_ids"], [])
        self.assertEqual(svc.corpus_version["metadata_release_ids"], ["R-2026-09-01"])

    def test_row_stamp_disagreeing_with_chunks_is_surfaced(self):
        """The ingest column and the chunk metadata are both recorded, so a
        disagreement is visible instead of one silently winning."""
        docs = [_make_doc("c1", "Học phí 2026", metadata={"release_id": "R-2026-09-01"})]
        svc = HybridSearchService(_make_retriever(docs, release_id="legacy"), _make_settings())
        svc.build_bm25_index()
        version = svc.corpus_version
        self.assertEqual(version["release_ids"], ["legacy"])
        self.assertEqual(version["metadata_release_ids"], ["R-2026-09-01"])


class TestRrfFusion(unittest.TestCase):
    def test_rrf_fusion_correctness(self):
        dense = [_make_doc("c1", score=0.9), _make_doc("c2", score=0.7)]
        sparse = [_make_doc("c2", score=5.0), _make_doc("c3", score=3.0)]
        retriever = _make_retriever()
        svc = HybridSearchService(retriever, _make_settings())
        fused = svc._rrf_fusion(dense, sparse, top_k=3)
        # c2 appears in both → should have higher fused score
        chunk_ids = [d.chunk_id for d in fused]
        self.assertIn("c2", chunk_ids)
        c2_idx = chunk_ids.index("c2")
        c2_score = fused[c2_idx].score
        self.assertGreater(c2_score, 0)
        # Deduplication: c2 must appear exactly once even though it was in both lists
        self.assertEqual(chunk_ids.count("c2"), 1)

    def test_rrf_fusion_weights(self):
        """dense_w=1.0, sparse_w=0.0 → ranking must match dense-only input order."""
        dense = [_make_doc("c1", score=0.9), _make_doc("c2", score=0.7)]
        sparse = [_make_doc("c2", score=5.0), _make_doc("c3", score=3.0)]
        retriever = _make_retriever()
        svc = HybridSearchService(retriever, _make_settings(dense_w=1.0, sparse_w=0.0))
        fused = svc._rrf_fusion(dense, sparse, top_k=5)
        chunk_ids = [d.chunk_id for d in fused]
        # With sparse weight=0, only dense ranks matter: c1 (rank 1) > c2 (rank 2)
        self.assertIn("c1", chunk_ids)
        self.assertIn("c2", chunk_ids)
        self.assertLess(chunk_ids.index("c1"), chunk_ids.index("c2"))


class TestSparseArmObeysPostFetchFilter(unittest.TestCase):
    """Ticket 09 criterion 4: the BM25 arm must be filtered by the SAME rule as
    the dense arm. Otherwise a stale chunk the dense arm hard-dropped (prior
    year, superseded) re-enters the fused pool through RRF — the sparse arm
    only demoted it, never dropped it."""

    def test_stale_bm25_candidate_cannot_re_enter_fused_pool(self):
        # BM25 needs >2 docs: rank_bm25's IDF is log((N-freq+0.5)/(freq+0.5)),
        # which is <= 0 when a term appears in 1 of only 2 documents — the
        # sparse arm would then return nothing and this test would pass
        # vacuously. The fillers give the query terms a positive IDF.
        docs = [
            _make_doc("keep", "điểm chuẩn ngành luật năm 2026"),
            _make_doc("stale", "học phí công nghệ thông tin mức thu theo tín chỉ"),
            _make_doc("filler1", "thời gian nộp hồ sơ xét tuyển"),
            _make_doc("filler2", "ký túc xá sinh viên nội trú"),
        ]
        retriever = _make_retriever(docs)
        # The dense arm hard-drops the stale chunk...
        retriever.retrieve.return_value = RetrievalResult(
            documents=[docs[0]], confidence=0.5
        )
        # ...and the shared post-fetch rule must drop it for the sparse arm too.
        retriever.apply_post_fetch_filters.side_effect = (
            lambda rows, *a, **k: [r for r in rows if r["chunk_id"] == "keep"]
        )

        svc = HybridSearchService(retriever, _make_settings())
        svc.build_bm25_index()
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                svc.retrieve(
                    query="học phí công nghệ thông tin",
                    query_vector=[0.1] * 10,
                    top_k=5,
                )
            )
        finally:
            loop.close()

        ids = [d.chunk_id for d in result.documents]
        self.assertIn("keep", ids)
        self.assertNotIn(
            "stale",
            ids,
            "BM25 candidates must obey the same post-fetch filter as dense",
        )


class TestDenseOnlyWhenFlagFalse(unittest.TestCase):
    """Ticket 09 criterion 5: turning hybrid off must be a config-only rollback.

    The observable contract is ``main.build_hybrid_service`` — the ONE place
    that decides whether a hybrid service exists. The earlier version of this
    test asserted ``settings.USE_HYBRID_RETRIEVAL is False`` on a MagicMock it
    had just configured with ``use_hybrid=False``: a tautology that passed
    whatever ``build_hybrid_service`` did, including if the flag were ignored
    entirely. This one calls the real function.
    """

    @staticmethod
    def _real_retriever():
        """A real LanceDBRetriever over a temp table, so the BM25 build is real."""
        import tempfile
        from pathlib import Path

        import lancedb
        import pyarrow as pa

        from services.lancedb_retrieval import LanceDBRetriever, LanceDBRetrieverConfig

        tmp = Path(tempfile.mkdtemp(prefix="hybrid_flag_"))
        table = lancedb.connect(str(tmp / "db")).create_table(
            "husc_chunks",
            data=[
                {
                    "id": "r1",
                    "chunk_id": "c1",
                    "text": "điểm chuẩn ngành luật năm 2026",
                    "source": "s",
                    "metadata_json": "{}",
                    "release_id": "R1",
                    "vector": [0.0] * 8,
                },
                {
                    "id": "r2",
                    "chunk_id": "c2",
                    "text": "học phí công nghệ thông tin",
                    "source": "s",
                    "metadata_json": "{}",
                    "release_id": "R1",
                    "vector": [0.0] * 8,
                },
                {
                    "id": "r3",
                    "chunk_id": "c3",
                    "text": "thời gian nộp hồ sơ xét tuyển",
                    "source": "s",
                    "metadata_json": "{}",
                    "release_id": "R1",
                    "vector": [0.0] * 8,
                },
            ],
        )
        config = LanceDBRetrieverConfig(
            uri=str(tmp / "db"), table_name="husc_chunks", embedding_dim=8
        )
        retriever = LanceDBRetriever(config)
        retriever._adapter._table = table
        return retriever

    def test_flag_off_returns_no_hybrid_service(self):
        """USE_HYBRID_RETRIEVAL=False → no service, so the query path stays dense-only."""
        import main as main_mod
        from config.settings import RAGSettings

        settings = RAGSettings()
        settings.USE_HYBRID_RETRIEVAL = False
        original = main_mod.settings
        main_mod.settings = settings
        try:
            self.assertIsNone(
                main_mod.build_hybrid_service(self._real_retriever()),
                "flag off must yield no hybrid service — otherwise flipping the "
                "flag back is not a rollback",
            )
        finally:
            main_mod.settings = original

    def test_flag_on_returns_hybrid_service(self):
        """The same call with the flag on DOES build one — so the test above
        cannot pass merely because build_hybrid_service is broken."""
        import main as main_mod
        from config.settings import RAGSettings

        settings = RAGSettings()
        settings.USE_HYBRID_RETRIEVAL = True
        original = main_mod.settings
        main_mod.settings = settings
        try:
            self.assertIsInstance(
                main_mod.build_hybrid_service(self._real_retriever()),
                HybridSearchService,
            )
        finally:
            main_mod.settings = original


class TestTombstoneBm25Index(unittest.TestCase):
    """T15(b): a tombstoned row must not be in the built BM25 corpus."""

    def _build_service(self, df) -> HybridSearchService:
        retriever = MagicMock()
        retriever.retrieve.return_value = RetrievalResult(documents=[], confidence=0.0)
        retriever.apply_post_fetch_filters.side_effect = lambda rows, *a, **k: rows

        table = MagicMock()
        # Include is_tombstone in the schema so it lands in load_cols
        table.schema.names = [
            "text", "source", "chunk_id", "metadata_json", "id", "release_id", "is_tombstone",
        ]
        select_result = MagicMock()
        select_result.to_pandas.return_value = df
        table.select.return_value = select_result
        table.to_pandas.return_value = df
        retriever._adapter._table = table

        svc = HybridSearchService(retriever, _make_settings())
        self.assertTrue(svc.build_bm25_index())
        return svc

    def test_tombstoned_row_not_in_bm25_corpus(self):
        df = pd.DataFrame(
            {
                "text": ["normal chunk", "tombstoned chunk"],
                "source": ["src1", "src2"],
                "chunk_id": ["ok", "bad"],
                "metadata_json": [
                    json.dumps({"is_superseded": False}),
                    json.dumps({"is_superseded": False}),
                ],
                "release_id": ["R1", "R1"],
                "is_tombstone": [False, True],
            }
        )
        svc = self._build_service(df)
        corpus_ids = [d.chunk_id for d in svc._corpus_docs]
        self.assertIn("ok", corpus_ids)
        self.assertNotIn("bad", corpus_ids, (
            "tombstoned row must not be in the BM25 corpus; "
            f"got {corpus_ids}"
        ))

    def test_null_tombstone_row_still_in_bm25_corpus(self):
        """The nullable-column trap: a row with is_tombstone=None must
        still be indexed — NULL means 'not tombstoned', not 'unknown'."""
        df = pd.DataFrame(
            {
                "text": ["un-stamped chunk"],
                "source": ["src"],
                "chunk_id": ["null1"],
                "metadata_json": [json.dumps({"is_superseded": False})],
                "release_id": ["R1"],
                "is_tombstone": [None],
            }
        )
        svc = self._build_service(df)
        corpus_ids = [d.chunk_id for d in svc._corpus_docs]
        self.assertIn("null1", corpus_ids, (
            "a NULL is_tombstone row is a legitimate chunk and must be "
            f"indexed; got {corpus_ids}"
        ))

    def test_metadata_mirror_tombstone_also_excluded(self):
        """When the top-level column is NULL but metadata_json carries
        is_tombstone=true, the row is still excluded — and when it is the
        ONLY row, the index build reports 'no indexable chunks' (False)."""
        df = pd.DataFrame(
            {
                "text": ["mirror-tombstone", "normal chunk"],
                "source": ["src", "src"],
                "chunk_id": ["mirror1", "ok"],
                "metadata_json": [
                    json.dumps({"is_superseded": False, "is_tombstone": True}),
                    json.dumps({"is_superseded": False}),
                ],
                "release_id": ["R1", "R1"],
                "is_tombstone": [None, False],  # column not stamped / not tombstone
            }
        )
        svc = self._build_service(df)
        corpus_ids = [d.chunk_id for d in svc._corpus_docs]
        self.assertNotIn("mirror1", corpus_ids, (
            "metadata mirror is_tombstone=true must exclude the row; "
            f"got {corpus_ids}"
        ))
        self.assertIn("ok", corpus_ids)
