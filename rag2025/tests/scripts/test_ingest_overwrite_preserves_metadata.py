"""T1.5-bl: full-overwrite rebuild must PRESERVE source-specific metadata.

The rebuild path (`ingest_lancedb.py` non-incremental) reads each JSONL row's
`metadata` dict and stores it verbatim into `metadata_json`. Fields like
`tuition_*`, `approved_by`, `chunk_method="manual_curated"`,
`fallback_template`, `year_mismatch_safe`, `image_url`, `parent_html`,
`thongbao_id` are carried by the `metadata` passthrough and must survive the
overwrite. This pins that contract so a future edit cannot strip them.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import lancedb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    d = Path("D:/tmp")
    d.mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture()
def temp_db(tmp_workspace: Path):
    d = tempfile.mkdtemp(dir="D:/tmp", prefix="test_meta_survive_")
    db = lancedb.connect(d)
    yield db
    try:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
    except OSError:
        pass


def _write_jsonl(path: Path, chunks: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks),
        encoding="utf-8",
    )


def _patch_ingest(mod, monkeypatch, temp_db, chunk_file, table_name="husc_chunks"):
    monkeypatch.setattr(mod, "MANIFEST_PATH", Path(tempfile.mktemp(dir="D:/tmp", suffix=".json")))
    monkeypatch.setattr(mod, "TARGET_TABLE", table_name)
    monkeypatch.setattr(mod, "_trigger_graph_sync", lambda *_: None)

    def _fake_resolve(chunks_dir, incremental):
        return [chunk_file]

    monkeypatch.setattr(mod, "_resolve_target_files", _fake_resolve)
    monkeypatch.setattr(mod, "_refresh_manifest_for_all_files", lambda *_: None)

    class FakeSettings:
        LANCEDB_URI = temp_db.uri
        LANCEDB_TABLE = table_name
        LANCEDB_UNIFIED_TABLE = table_name
        EMBEDDING_MODEL = "fake"
        EMBEDDING_DIM = 1024

    monkeypatch.setattr(mod, "RAGSettings", FakeSettings)

    vec_fixed = np.full(1024, 0.1)

    class FakeEmbeddingService:
        def __init__(self, settings):
            pass

        def encode_documents(self, texts):
            return [vec_fixed.copy() for _ in texts]

    monkeypatch.setattr(mod, "EmbeddingService", FakeEmbeddingService)


CANONICAL_METADATA = {
    "approved_by": "khôn_khéo_pattern_v1",
    "chunk_method": "manual_curated",
    "data_year": 2026,
    "fallback_template": True,
    "year_mismatch_safe": True,
    "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=73",
}


def test_overwrite_preserves_special_metadata(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """Full overwrite (incremental=False) keeps tuition_*, approved_by,
    chunk_method, fallback_template, year_mismatch_safe in metadata_json."""
    import rag2025.scripts.ingest_lancedb as mod

    # Pre-existing table (simulates production husc_chunks) so overwrite path runs.
    temp_db.create_table(
        "husc_chunks",
        data=[{"id": "old1", "text": "old row", "vector": [0.1] * 1024}],
    )

    chunk_file = tmp_workspace / "husc_2026_canonical_v3.jsonl"
    _write_jsonl(chunk_file, [
        {
            "id": "canon1",
            "text": "Thông tin tuyển sinh Đại học Khoa học 2026",
            "metadata": {
                **CANONICAL_METADATA,
                "tuition_hoc_phi_vnd_per_credit": 350000,
                "tuition_khoi_IV": "1.200.000 VNĐ/tháng",
            },
        },
        {
            "id": "canon2",
            "text": "Chỉ tiêu tuyển sinh năm 2026 các ngành đào tạo",
            "metadata": {
                **CANONICAL_METADATA,
                "image_url": "https://tuyensinh.husc.edu.vn/img/ct.png",
                "parent_html": "<div><p>parent</p></div>",
                "thongbao_id": 73,
            },
        },
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=False)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    assert table.count_rows() == 2

    rows = {r["id"]: json.loads(r["metadata_json"]) for r in table.to_arrow().to_pylist()}
    assert len(rows) == 2

    # Row canon1: tuition_*, approved_by, chunk_method, fallback, year_mismatch
    m1 = next(m for rid, m in rows.items() if m.get("source_url", "").endswith("id=73"))
    assert m1["approved_by"] == "khôn_khéo_pattern_v1"
    assert m1["chunk_method"] == "manual_curated"
    assert m1["fallback_template"] is True
    assert m1["year_mismatch_safe"] is True
    assert m1["tuition_hoc_phi_vnd_per_credit"] == 350000
    assert m1["tuition_khoi_IV"] == "1.200.000 VNĐ/tháng"
    # ingest_lancedb adds provenance + guardrails, must not clobber source fields
    assert m1["ingest_file"] == chunk_file.name

    # Row canon2: image_url, parent_html, thongbao_id survive
    m2 = next(m for rid, m in rows.items() if m.get("thongbao_id") == 73)
    assert m2["image_url"] == "https://tuyensinh.husc.edu.vn/img/ct.png"
    assert m2["parent_html"] == "<div><p>parent</p></div>"
    assert m2["thongbao_id"] == 73
    assert m2["chunk_method"] == "manual_curated"


def test_overwrite_preserves_vision_ocr_metadata(
    tmp_workspace: Path, temp_db: Any, monkeypatch: Any
):
    """vision_ocr rows keep image_url/parent_html/thongbao_id through overwrite."""
    import rag2025.scripts.ingest_lancedb as mod

    temp_db.create_table(
        "husc_chunks",
        data=[{"id": "old1", "text": "old row", "vector": [0.1] * 1024}],
    )

    chunk_file = tmp_workspace / "husc_2026_vision_ocr.jsonl"
    _write_jsonl(chunk_file, [
        {
            "id": "ocr1",
            "text": "Ảnh OCR thông báo tuyển sinh số 65",
            "metadata": {
                "approved_by": "vision_pipeline_v1",
                "chunk_method": "vision_ocr_claude_sonnet_4_6",
                "data_year": 2026,
                "image_url": "https://tuyensinh.husc.edu.vn/upload/65.png",
                "parent_html": "<img src='65.png'/>",
                "thongbao_id": 65,
                "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=65",
            },
        },
    ])

    _patch_ingest(mod, monkeypatch, temp_db, chunk_file)

    result = mod.ingest_lancedb(incremental=False)
    assert result is True

    table = temp_db.open_table("husc_chunks")
    rows = table.to_arrow().to_pylist()
    assert len(rows) == 1

    m = json.loads(rows[0]["metadata_json"])
    assert m["chunk_method"] == "vision_ocr_claude_sonnet_4_6"
    assert m["approved_by"] == "vision_pipeline_v1"
    assert m["image_url"] == "https://tuyensinh.husc.edu.vn/upload/65.png"
    assert m["parent_html"] == "<img src='65.png'/>"
    assert m["thongbao_id"] == 65
