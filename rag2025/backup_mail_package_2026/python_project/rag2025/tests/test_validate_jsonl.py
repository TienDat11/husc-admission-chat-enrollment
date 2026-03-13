"""
Unit tests for validate_jsonl module (Epic 1.1)
"""
import json
import tempfile
from pathlib import Path

import pytest

from src.validate_jsonl import (
    ValidationReport,
    detect_encoding,
    load_schema,
    validate_jsonl,
)


@pytest.fixture
def schema_path():
    """Fixture for JSON Schema path."""
    return Path(__file__).parent.parent / "config" / "rag_chunk_schema.json"


@pytest.fixture
def valid_record():
    """Fixture for valid record."""
    return {
        "id": "test_001",
        "doc_id": "doc_001",
        "chunk_id": 0,
        "text": "This is a test chunk with sufficient length for validation.",
        "metadata": {"source": "test_source", "year": 2025, "expired": False},
    }


@pytest.fixture
def invalid_record():
    """Fixture for invalid record (missing required fields)."""
    return {
        "id": "test_002",
        "text": "Missing doc_id and metadata fields",
    }


def test_load_schema(schema_path):
    """Test schema loading."""
    schema = load_schema(schema_path)

    assert isinstance(schema, dict)
    assert schema["title"] == "RAGChunk"
    assert "required" in schema
    assert "id" in schema["required"]


def test_detect_encoding():
    """Test encoding detection."""
    # Create UTF-8 test file
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".txt") as f:
        f.write("Test content with UTF-8 encoding: 你好世界")
        temp_path = Path(f.name)

    try:
        encoding = detect_encoding(temp_path)
        assert encoding in ["utf-8", "UTF-8", "ascii"]
    finally:
        temp_path.unlink()


def test_validate_jsonl_valid_records(schema_path, valid_record):
    """Test validation with valid records."""
    # Create temp input file
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".jsonl"
    ) as f:
        f.write(json.dumps(valid_record) + "\n")
        f.write(json.dumps(valid_record) + "\n")  # Write twice
        input_path = Path(f.name)

    # Create temp output file path
    output_path = input_path.parent / "validated_output.jsonl"

    try:
        report = validate_jsonl(input_path, schema_path, output_path, strict=True)

        assert report.total_lines == 2
        assert report.valid_lines == 2
        assert report.invalid_lines == 0
        assert report.success_rate == 1.0
        assert len(report.errors) == 0
        assert output_path.exists()

    finally:
        input_path.unlink()
        if output_path.exists():
            output_path.unlink()


def test_validate_jsonl_invalid_records(schema_path, invalid_record):
    """Test validation with invalid records."""
    # Create temp input file
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".jsonl"
    ) as f:
        f.write(json.dumps(invalid_record) + "\n")
        input_path = Path(f.name)

    try:
        report = validate_jsonl(input_path, schema_path, output_path=None, strict=True)

        assert report.total_lines == 1
        assert report.invalid_lines == 1
        assert report.valid_lines == 0
        assert len(report.errors) > 0
        assert report.errors[0]["error"] == "Schema validation failed"

    finally:
        input_path.unlink()


def test_validate_jsonl_malformed_json(schema_path):
    """Test validation with malformed JSON."""
    # Create temp input file
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".jsonl"
    ) as f:
        f.write('{"id": "test", incomplete json\n')  # Malformed
        input_path = Path(f.name)

    try:
        report = validate_jsonl(input_path, schema_path, output_path=None, strict=True)

        assert report.invalid_lines >= 1
        assert len(report.errors) > 0
        assert any(err["error"] == "JSON parse error" for err in report.errors)

    finally:
        input_path.unlink()


def test_validate_jsonl_empty_lines(schema_path, valid_record):
    """Test validation with empty lines."""
    # Create temp input file
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".jsonl"
    ) as f:
        f.write(json.dumps(valid_record) + "\n")
        f.write("\n")  # Empty line
        f.write(json.dumps(valid_record) + "\n")
        input_path = Path(f.name)

    try:
        report = validate_jsonl(input_path, schema_path, output_path=None, strict=True)

        assert report.total_lines == 3
        assert report.valid_lines == 2
        assert len(report.warnings) > 0
        assert any("Empty line" in w for w in report.warnings)

    finally:
        input_path.unlink()


def test_validation_report_summary():
    """Test ValidationReport summary generation."""
    report = ValidationReport(
        total_lines=100,
        valid_lines=95,
        invalid_lines=5,
        encoding="utf-8",
        errors=[],
        warnings=["Warning 1"],
    )

    summary = report.summary()
    assert "Total: 100" in summary
    assert "Valid: 95" in summary
    assert "Success: 95.0%" in summary
    assert "Encoding: utf-8" in summary


def test_vector_dimension_validation(schema_path):
    """Test vector dimension validation (must be 768)."""
    record_wrong_dim = {
        "id": "test_003",
        "doc_id": "doc_003",
        "chunk_id": 0,
        "text": "Test chunk with wrong vector dimension",
        "vector": [0.1] * 512,  # Wrong dimension (512 instead of 768)
        "metadata": {"source": "test_source"},
    }

    # Create temp input file
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, suffix=".jsonl"
    ) as f:
        f.write(json.dumps(record_wrong_dim) + "\n")
        input_path = Path(f.name)

    try:
        report = validate_jsonl(input_path, schema_path, output_path=None, strict=True)

        assert report.invalid_lines >= 1
        # Schema should catch wrong vector dimension

    finally:
        input_path.unlink()
