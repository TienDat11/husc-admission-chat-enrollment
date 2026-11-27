"""Tests for src.config.paths: locations derive from the package, not the cwd."""
from pathlib import Path

from src.config import paths


def test_root_is_rag2025_package_root():
    assert paths.RAG2025_ROOT == Path(__file__).resolve().parents[2]
    assert (paths.RAG2025_ROOT / "src" / "config" / "paths.py").is_file()


def test_data_dirs_hang_off_data_dir():
    assert paths.DATA_DIR == paths.RAG2025_ROOT / "data"
    assert paths.MAJOR_CODES_DIR == paths.DATA_DIR / "major_codes"
    assert paths.CANONICAL_CORPUS_DIR == paths.DATA_DIR / "canonical_corpus"
    assert paths.YEAR_FACTS_DIR == paths.DATA_DIR / "year_facts"
    assert paths.CHUNKED_DIR == paths.DATA_DIR / "chunked"
    assert paths.LANCEDB_DIR == paths.DATA_DIR / "lancedb"


def test_paths_are_absolute():
    assert paths.DATA_DIR.is_absolute()
