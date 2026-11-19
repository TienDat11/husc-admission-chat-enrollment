"""U8: retrieval diff sandbox — before/after top-k, entered/dropped, rank_delta."""

from __future__ import annotations

from src.services.retrieval_sandbox import (
    Chunk,
    Query,
    SandboxResult,
    compute_diff,
    tokenize,
)
from src.services.text_stopwords import STOPWORDS_VI


def test_tokenize_drops_stopwords():
    toks = tokenize("Điểm chuẩn ngành công nghệ thông tin năm 2026")
    assert "chuẩn" in toks
    assert "công" in toks
    assert "nghệ" in toks
    assert "thông" in toks
    assert "tin" in toks
    assert "2026" in toks
    # stopwords removed
    assert "là" not in toks
    assert "của" not in toks
    assert "và" not in toks


def test_compute_diff_identical_before_after():
    chunks = [
        Chunk(chunk_id="c1", text="a b c", record_id="r1", embedding=[0.1, 0.2]),
        Chunk(chunk_id="c2", text="d e f", record_id="r1", embedding=[0.3, 0.4]),
    ]
    query = Query(query_id="q1", text="a b c", embedding=[0.1, 0.2])
    result: SandboxResult = compute_diff(
        queries=[query],
        before_chunks=list(chunks),
        after_chunks=list(chunks),
        k=5,
    )
    assert result.status == "ok"
    qd = result.queries[0]
    assert set(qd.before_top_k) == {"c1", "c2"}
    assert set(qd.after_top_k) == {"c1", "c2"}
    assert qd.entered == []
    assert qd.left == []
    assert all(d == 0 for d in qd.rank_delta.values())


def test_compute_diff_entered_and_dropped():
    before = [
        Chunk(chunk_id="c1", text="old chunk", record_id="r1", embedding=[1.0, 0.0]),
    ]
    after = [
        Chunk(chunk_id="c1", text="old chunk", record_id="r1", embedding=[1.0, 0.0]),
        Chunk(chunk_id="c2", text="new chunk", record_id="r1", embedding=[0.0, 1.0]),
    ]
    query = Query(query_id="q1", text="new chunk", embedding=[0.0, 1.0])
    result = compute_diff(
        queries=[query],
        before_chunks=before,
        after_chunks=after,
        k=5,
    )
    qd = result.queries[0]
    assert "c2" in qd.after_top_k
    assert "c2" in qd.entered


def test_empty_queries():
    result = compute_diff(queries=[], before_chunks=[], after_chunks=[], k=5)
    assert result.status == "ok"
    assert result.queries == []


def test_stopwords_are_frozenset():
    assert isinstance(STOPWORDS_VI, frozenset)
    assert len(STOPWORDS_VI) > 0
