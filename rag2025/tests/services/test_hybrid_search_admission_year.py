"""HybridSearchService re-reads the admission year on use, not at import."""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import services.lancedb_retrieval as lr  # noqa: E402
from services.hybrid_search import HybridSearchService  # noqa: E402


def test_temporal_boost_follows_invalidated_admission_year(monkeypatch):
    years = iter([2030, 2031])
    monkeypatch.setattr(lr, "_detect_current_admission_year", lambda: next(years))
    lr.invalidate_admission_year_cache()
    try:
        doc = lr.RetrievedDocument(text="x", metadata={"data_year": 2030})
        boost = HybridSearchService._apply_temporal_bm25_boost
        assert boost(doc, 2.0) == 2.0 * lr.YEAR_BOOST_FACTOR

        # Table swap: the year moves on, so the same chunk is now historical.
        lr.invalidate_admission_year_cache()
        assert boost(doc, 2.0) == 2.0 * lr.OLD_YEAR_DEMOTE_FACTOR
    finally:
        lr.invalidate_admission_year_cache()
