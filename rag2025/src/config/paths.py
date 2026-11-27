"""Filesystem locations derived once from the package layout.

Derived from this file's location, so they do not depend on the process cwd.
"""
from pathlib import Path

RAG2025_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = RAG2025_ROOT / "data"
MAJOR_CODES_DIR = DATA_DIR / "major_codes"
CANONICAL_CORPUS_DIR = DATA_DIR / "canonical_corpus"
YEAR_FACTS_DIR = DATA_DIR / "year_facts"
CHUNKED_DIR = DATA_DIR / "chunked"
LANCEDB_DIR = DATA_DIR / "lancedb"
GRAPH_DIR = DATA_DIR / "graph"
KNOWLEDGE_GRAPH_PATH = GRAPH_DIR / "knowledge_graph.graphml"
EVAL_DIR = DATA_DIR / "eval"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
CANONICAL_RECORDS_DIR = DATA_DIR / "canonical_records"
AUDIT_PRE_REINGEST_PATH = DATA_DIR / "audit_pre_reingest.json"
MAJOR_HISTORY_PATH = MAJOR_CODES_DIR / "diem_chuan_history.json"
MAJOR_SCOPE_FILE = MAJOR_CODES_DIR / "husc_majors_current.json"
TUITION_2026_OFFICIAL_FILE = MAJOR_CODES_DIR / "husc_tuition_2026_official.json"
