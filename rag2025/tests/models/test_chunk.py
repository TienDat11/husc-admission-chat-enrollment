"""Tests for ChunkMetadataV3 (TDD V5-R030 1:1 mapping)."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest
from pydantic import ValidationError

# Ensure rag2025/src is importable.
RAG_SRC = Path(__file__).resolve().parents[2] / "src"
if str(RAG_SRC) not in sys.path:
    sys.path.insert(0, str(RAG_SRC))

from models.chunk import ChunkMetadataV3  # noqa: E402


VALID_HASH = "a" * 64


def _strict_payload(**overrides):
    base = {
        "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=74",
        "notification_id": 74,
        "crawl_date": datetime(2026, 5, 25, tzinfo=timezone.utc),
        "data_year": 2026,
        "chunk_method": "rule_v2",
        "chunk_version_hash": VALID_HASH,
        "info_type": "diem_chuan",
    }
    base.update(overrides)
    return base


def test_strict_construction_succeeds_with_valid_payload():
    meta = ChunkMetadataV3(**_strict_payload())
    assert meta.data_year == 2026
    assert meta.chunk_method == "rule_v2"
    assert meta.school == "HUSC"
    assert meta.audience == "thi_sinh"


def test_strict_rejects_data_year_below_range():
    with pytest.raises(ValidationError):
        ChunkMetadataV3(**_strict_payload(data_year=2023))


def test_strict_rejects_data_year_above_range():
    with pytest.raises(ValidationError):
        ChunkMetadataV3(**_strict_payload(data_year=2031))


def test_strict_rejects_invalid_chunk_method():
    with pytest.raises(ValidationError):
        ChunkMetadataV3(**_strict_payload(chunk_method="random_v9"))


def test_strict_rejects_short_version_hash():
    with pytest.raises(ValidationError):
        ChunkMetadataV3(**_strict_payload(chunk_version_hash="abc"))


def test_strict_rejects_long_version_hash():
    with pytest.raises(ValidationError):
        ChunkMetadataV3(**_strict_payload(chunk_version_hash="a" * 65))


def test_from_legacy_with_full_fields():
    legacy = {
        "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=59",
        "notification_id": 59,
        "crawl_date": "2025-06-01T00:00:00+00:00",
        "data_year": 2025,
        "chunk_method": "rule_v2",
        "chunk_version_hash": "b" * 64,
        "info_type": "hoc_phi",
        "extra_legacy_field": "kept-as-passthrough",
    }
    meta = ChunkMetadataV3.from_legacy(legacy)
    assert meta.data_year == 2025
    assert meta.notification_id == 59
    assert meta.info_type == "hoc_phi"
    # Passthrough preserved via extra="allow"
    dumped = meta.model_dump()
    assert dumped.get("extra_legacy_field") == "kept-as-passthrough"


def test_from_legacy_uses_year_field_when_data_year_missing():
    legacy = {"year": 2025, "info_type": "qa"}
    meta = ChunkMetadataV3.from_legacy(legacy)
    assert meta.data_year == 2025


def test_from_legacy_falls_back_to_default_when_no_year():
    legacy = {"info_type": "qa"}
    meta = ChunkMetadataV3.from_legacy(legacy)
    # HIGH-3 fix: default year now comes from CURRENT_ADMISSION_YEAR env (default 2026)
    assert meta.data_year == 2026
    assert meta.chunk_method == "rule_v2"
    assert meta.chunk_version_hash == "0" * 64


def test_from_legacy_uses_defaults_dict_for_missing_source_url():
    legacy = {"info_type": "qa"}
    defaults = {"source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=74"}
    meta = ChunkMetadataV3.from_legacy(legacy, defaults)
    assert str(meta.source_url).startswith("https://tuyensinh.husc.edu.vn/thongbao.php?id=74")


def test_from_legacy_does_not_raise_on_minimal_input():
    # Truly minimal — only info_type provided
    meta = ChunkMetadataV3.from_legacy({"info_type": "unknown"})
    assert meta.school == "HUSC"
    assert meta.audience == "thi_sinh"
    assert meta.is_superseded is False


def test_from_legacy_round_trip_preserves_year():
    legacy = {"data_year": 2026, "info_type": "diem_chuan", "source_url": "https://tuyensinh.husc.edu.vn/thongbao.php?id=74"}
    meta1 = ChunkMetadataV3.from_legacy(legacy)
    dumped = meta1.model_dump()
    meta2 = ChunkMetadataV3.from_legacy(dumped)
    assert meta1.data_year == meta2.data_year
    assert meta1.info_type == meta2.info_type
