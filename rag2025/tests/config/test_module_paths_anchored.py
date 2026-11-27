"""Module-level data paths come from src/config/paths.py.

The constants must resolve to the files the replaced Path(__file__)
derivations pointed at, and the module paths must not depend on the cwd.
"""
import inspect
from pathlib import Path

from src.config import paths


def test_shared_paths_anchor_to_rag2025_data_dir():
    rag2025 = Path(paths.__file__).resolve().parents[2]
    data = rag2025 / "data"
    assert paths.RAG2025_ROOT == rag2025
    assert paths.DATA_DIR == data
    assert paths.KNOWLEDGE_GRAPH_PATH == data / "graph" / "knowledge_graph.graphml"
    assert paths.EVAL_DIR == data / "eval"
    assert paths.SNAPSHOTS_DIR == data / "snapshots"
    assert paths.CANONICAL_RECORDS_DIR == data / "canonical_records"
    assert paths.AUDIT_PRE_REINGEST_PATH == data / "audit_pre_reingest.json"
    assert paths.MAJOR_HISTORY_PATH == data / "major_codes" / "diem_chuan_history.json"
    assert paths.MAJOR_SCOPE_FILE == data / "major_codes" / "husc_majors_current.json"
    assert paths.TUITION_2026_OFFICIAL_FILE == data / "major_codes" / "husc_tuition_2026_official.json"


def test_module_constants_use_the_shared_paths():
    from src.infrastructure import auto_crawl, win32_utils
    from src.services import graphrag_retriever, guardrail, major_recommender

    assert graphrag_retriever.GRAPH_PATH == paths.KNOWLEDGE_GRAPH_PATH
    assert major_recommender._HISTORY_PATH == paths.MAJOR_HISTORY_PATH

    owners = [
        value
        for value in vars(guardrail).values()
        if isinstance(value, type) and "MAJOR_SCOPE_FILE" in vars(value)
    ]
    assert len(owners) == 1
    assert owners[0].MAJOR_SCOPE_FILE == paths.MAJOR_SCOPE_FILE
    assert owners[0]._MAJOR_FALLBACK_FILES == (paths.TUITION_2026_OFFICIAL_FILE,)

    assert auto_crawl.SNAPSHOT_BASE == str(paths.SNAPSHOTS_DIR)
    assert auto_crawl.DATA_DIRS == [
        str(paths.CHUNKED_DIR),
        str(paths.LANCEDB_DIR),
        str(paths.GRAPH_DIR),
    ]

    snapshot_default = inspect.signature(win32_utils.cleanup_old_snapshots).parameters["base_dir"].default
    journal_default = inspect.signature(win32_utils.check_stale_journals).parameters["db_path"].default
    assert snapshot_default == paths.SNAPSHOTS_DIR
    assert journal_default == paths.LANCEDB_DIR


def test_golden_and_audit_paths_do_not_follow_cwd(monkeypatch, tmp_path):
    from src.routers import meta
    from src.services import golden_retrieval_check

    monkeypatch.chdir(tmp_path)
    assert golden_retrieval_check.golden_set_path() == (
        paths.EVAL_DIR / golden_retrieval_check.GOLDEN_RECORDS_FILENAME
    )
    assert meta._DEFAULT_AUDIT_PATH == paths.AUDIT_PRE_REINGEST_PATH
