"""Tests for crawl_incremental.chunk_fn adapter."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from crawl_incremental import chunk_fn


@pytest.fixture()
def out_dir(tmp_path: Path) -> Path:
    """Force chunk_fn to write to tmp_path — regression: the adapter wrote
    556-byte crawl stubs (chunked_thongbao_78_*.jsonl) into the real
    data/chunked/ on every pytest run, breaking test_graph_rebuild_parity."""
    return tmp_path


def _sample_text():
    return (
        "CHƯƠNG TRÌNH TUYỂN SINH 2026\n\n"
        "1. Điều kiện xét tuyển\n"
        "Thí sinh tốt nghiệp THPT hoặc tương đương.\n\n"
        "2. Các phương thức xét tuyển\n"
        "Phương thức 1: Xét học bạ THPT\n"
        "Phương thức 2: Xét kết quả thi THPT\n\n"
        "3. Chỉ tiêu招生\n"
        "Ngành Kỹ thuật phần mềm: 120 chỉ tiêu\n\n"
        "4. Học phí\n"
        "4.500.000 VNĐ/tín chỉ."
    )


def test_chunk_fn_returns_non_empty(out_dir):
    chunks = chunk_fn(_sample_text(), {"source_url": "https://example.com?id=78"}, chunked_dir=out_dir)
    assert len(chunks) > 0, "chunk_fn must return at least one chunk"


def test_chunk_fn_has_required_keys(out_dir):
    chunks = chunk_fn(_sample_text(), {"source_url": "https://example.com?id=78"}, chunked_dir=out_dir)
    for c in chunks:
        assert "id" in c, "missing 'id'"
        assert "chunk_id" in c, "missing 'chunk_id'"
        assert "text" in c, "missing 'text'"
        assert "metadata" in c, "missing 'metadata'"
        assert isinstance(c["text"], str), "text must be string"
        assert len(c["text"]) > 0, "text must not be empty"


def test_chunk_fn_parses_notification_id(out_dir):
    chunks = chunk_fn(_sample_text(), {"source_url": "https://example.com?id=78"}, chunked_dir=out_dir)
    for c in chunks:
        meta = c["metadata"]
        assert meta.get("notification_id") == 78, f"expected nid=78, got {meta.get('notification_id')}"
        assert "crawl_date" in meta, "missing crawl_date"
        assert c["id"].startswith("thongbao_78"), f"id prefix wrong: {c['id']}"


def test_chunk_fn_no_url_id(out_dir):
    chunks = chunk_fn(_sample_text(), {"source_url": "https://example.com/page"}, chunked_dir=out_dir)
    for c in chunks:
        assert c["id"].startswith("thongbao_unknown"), f"expected thongbao_unknown prefix: {c['id']}"


def test_chunk_fn_max_chars_limit(out_dir):
    long_text = "\n\n".join([f"Paragraph {i} with some content to exceed 1200 chars limit eventually when joined together properly." for i in range(50)])
    chunks = chunk_fn(long_text, {"source_url": "https://example.com?id=99"}, chunked_dir=out_dir)
    # Should produce multiple chunks since text is long
    assert len(chunks) >= 1
    # Each chunk should be <= ~1200 chars + overhead
    for c in chunks:
        assert len(c["text"]) <= 1500, f"chunk too large: {len(c['text'])} chars"
