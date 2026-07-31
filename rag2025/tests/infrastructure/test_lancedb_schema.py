"""Tests for lancedb_schema + migration (M3-B1). TDD V5-R030."""
import importlib.util, sys, subprocess, tempfile
from pathlib import Path
import pyarrow as pa
import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "infrastructure"
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def schema_module():
    spec = importlib.util.spec_from_file_location(
        "infrastructure.lancedb_schema", SRC / "lancedb_schema.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_schema_exact(schema_module):
    """CHUNK_SCHEMA khớp 17 field, dim=1024."""
    s = schema_module.CHUNK_SCHEMA
    assert len(s) == 17
    names = [f.name for f in s]
    assert "id" in names
    assert "vector" in names
    assert "superseded_by" in names
    assert "data_year" in names
    assert "is_reference" in names
    assert "crawl_date" in names
    # vector dim
    vec_field = s.field("vector")
    assert vec_field.type.list_size == 1024


def test_schema_nullable_matches_lancedb_reality(schema_module):
    """7 field từng nullable=False phải là True — LanceDB coerce mọi field thành nullable."""
    s = schema_module.CHUNK_SCHEMA
    nullable_fields = ["release_id", "scope", "doc_type", "status", "effective_from", "effective_to", "is_tombstone"]
    for fname in nullable_fields:
        f = s.field(fname)
        assert f.nullable is True, f"{fname} must be nullable=True (LanceDB reality)"


def test_schema_has_is_reference_and_crawl_date(schema_module):
    """CHUNK_SCHEMA phải có is_reference(bool) và crawl_date(string)."""
    s = schema_module.CHUNK_SCHEMA
    names = [f.name for f in s]
    assert "is_reference" in names
    ir = s.field("is_reference")
    assert ir.type == pa.bool_()
    assert ir.nullable is True
    assert "crawl_date" in names
    cd = s.field("crawl_date")
    assert cd.type == pa.string()
    assert cd.nullable is True


def test_is_reference_in_bitmap_columns(schema_module):
    """is_reference phải nằm trong BITMAP_COLUMNS."""
    assert "is_reference" in schema_module.BITMAP_COLUMNS


def test_indices_defined(schema_module):
    """ALL_INDEX_COLUMNS >= 8."""
    assert len(schema_module.ALL_INDEX_COLUMNS) >= 8
    assert "id" in schema_module.BTREE_COLUMNS
    assert "status" in schema_module.BITMAP_COLUMNS


def test_env_dim_matches_table(schema_module):
    """EMBEDDING_DIM == 1024 (khớp .env production + bảng thật)."""
    assert schema_module.EMBEDDING_DIM == 1024


def test_migration_dry_run_no_write(schema_module, tmp_path):
    """--dry-run không tạo bảng mới."""
    import lancedb

    db_path = tmp_path / "lancedb"
    db = lancedb.connect(str(db_path))

    # Tạo bảng rag2025 giả (3 rows, schema thật)
    rows = [
        {
            "id": "a1",
            "chunk_id": "a1",
            "text": "hello",
            "source": "f.jsonl",
            "faq_type": "tuyen_sinh",
            "metadata_json": '{"data_year": 2026}',
            "vector": [0.1] * 1024,
        },
    ]
    db.create_table("rag2025", data=rows)
    db.create_table("husc_v2026_clean", data=[
        {
            "id": "h1",
            "text": "world",
            "summary": "s",
            "embedding": [0.2] * 1024,
            "metadata": {"data_year": 2026, "chunk_type": "nganh"},
        },
    ])

    # Chạy migration dry-run
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "migrate_schema_v2.py"),
            "--dry-run",
            "--uri", str(db_path),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "husc_chunks" not in db.table_names()  # dry-run không tạo
