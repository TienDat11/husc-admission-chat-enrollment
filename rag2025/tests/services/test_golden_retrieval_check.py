"""T06 (spec §02:87 lớp d rút gọn): golden-question top-5 check.

Zero-LLM retrieval gate measured two ways:

* ``run_golden_retrieval_check`` against a scripted retriever — the miss
  case names the exact golden question, the pass case reports
  "x/y câu hỏi mẫu vẫn tìm đúng thông tin", and the report is advisory
  only (it carries no blocking verdict anywhere in its shape);
* the HTTP seam — the review-preview response carries the report under
  ``golden_retrieval`` and stays 200 even with failures: cảnh báo,
  không chặn.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.canonical.diff import diff_records  # noqa: E402
from src.domain.canonical.record import CanonicalRecord  # noqa: E402
from src.routers.admin_preview import router as admin_preview_router  # noqa: E402
from src.services.golden_retrieval_check import (  # noqa: E402
    GoldenRetrievalReport,
    load_golden_set,
    run_golden_retrieval_check,
)
from src.infrastructure.canonical_record_store import (  # noqa: E402
    CanonicalRecordStore,
    RecordNotFoundError,
)
from src.services.lancedb_retrieval import (  # noqa: E402
    RetrievedDocument,
    RetrievalError,
    RetrievalResult,
)
from src.services.review_preview import ReviewPreviewResult  # noqa: E402

GOLDEN = [
    {
        "id": "q001",
        "question": "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026 là bao nhiêu?",
        "expected_record_ids": ["HOC_PHI:nganh:7480201:2026:canon-nganh"],
    },
    {
        "id": "q002",
        "question": "Mã ngành Kỹ thuật phần mềm là gì?",
        "expected_record_ids": ["NGANH:nganh:7480103:2026:canon-nganh"],
    },
]


class FakeEmbedder:
    def encode_query(self, text: str) -> list[float]:
        return [0.0]


class _ScriptedRetriever:
    """Duck-typed ``LanceDBRetriever``: ``retrieve(question_key)`` per question."""

    def __init__(self, top5_by_question: dict[str, list[str]]):
        self._top5 = top5_by_question

    def retrieve(self, *, query_vector, top_k=None, metadata_filter=None, query=None):
        record_ids = self._top5.get(str(query), [])
        docs = [
            RetrievedDocument(
                text=f"chunk {rid}",
                source="s",
                chunk_id=f"chunk-{rid}",
                metadata={"record_id": rid},
                score=0.9,
            )
            for rid in record_ids[:5]
        ]
        return RetrievalResult(
            documents=docs,
            error_type=None if docs else RetrievalError.EMPTY_RESULT,
        )


def _retriever_with(question: str, record_ids: list[str]) -> _ScriptedRetriever:
    return _ScriptedRetriever({question: record_ids})


# ---------------------------------------------------------------------------
# the check itself: miss names the question, pass counts, advisory shape
# ---------------------------------------------------------------------------




def test_miss_reports_names_the_missing_question_and_never_blocks():
    retriever = _ScriptedRetriever(
        {
            "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026 là bao nhiêu?": [
                "HOC_PHI:nganh:7480201:2026:canon-nganh",
            ],
            "Mã ngành Kỹ thuật phần mềm là gì?": [
                "HOC_PHI:nganh:7480103:2026:canon-nganh",  # wrong record for q002
            ],
        }
    )
    report = run_golden_retrieval_check(GOLDEN, retriever, FakeEmbedder())
    assert report.total == 2
    assert report.passed == 1
    assert len(report.failed) == 1
    failed = report.failed[0]
    assert failed["id"] == "q002"
    assert failed["question"].startswith("Mã ngành Kỹ thuật phần mềm")
    assert failed["expected_record_ids"] == ["NGANH:nganh:7480103:2026:canon-nganh"]
    # The report carries no blocking verdict of any kind — layer (d) is
    # advisory: the only fields are counts, the named misses, the UX line.
    body = report.to_dict()
    assert set(body) == {"passed", "total", "failed", "summary"}
    assert "⚠" in body["summary"]
    assert "q002" in body["summary"]
def test_all_pass_reports_the_green_count():
    retriever = _ScriptedRetriever(
        {
            "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026 là bao nhiêu?": [
                "HOC_PHI:nganh:7480201:2026:canon-nganh",
            ],
            "Mã ngành Kỹ thuật phần mềm là gì?": [
                "NGANH:nganh:7480103:2026:canon-nganh",
            ],
        }
    )
    report = run_golden_retrieval_check(GOLDEN, retriever, FakeEmbedder())
    assert report.failed == ()
    assert report.passed == 2 and report.total == 2
    # Spec §05:61 green line: "x/y câu hỏi mẫu vẫn tìm đúng thông tin".
    assert report.summary == "2/2 câu hỏi mẫu vẫn tìm đúng thông tin"


def test_expected_record_in_top5_by_any_expected_member_passes():
    # A question expecting two records passes when EITHER lands in top-5 —
    # the gate is "record mong đợi có trong top-5", not "all of them".
    golden = [
        {
            "id": "q9",
            "question": "q",
            "expected_record_ids": ["a", "b"],
        }
    ]
    retriever = _retriever_with("q", ["b", "x", "y"])
    assert run_golden_retrieval_check(golden, retriever, FakeEmbedder()).passed == 1


def test_empty_retrieval_counts_as_a_miss():
    retriever = _ScriptedRetriever({})
    report = run_golden_retrieval_check(GOLDEN, retriever, FakeEmbedder())
    assert report.passed == 0
    assert [f["id"] for f in report.failed] == ["q001", "q002"]


def test_golden_set_loader_reads_the_repo_artifact():
    questions = load_golden_set()
    assert questions, "the golden artifact must carry at least one expected set"
    for q in questions:
        assert q["question"] and q["expected_record_ids"]
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids))


def test_golden_set_loader_returns_empty_without_the_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.services.golden_retrieval_check.golden_set_path",
        lambda: tmp_path / "missing.json",
    )
    assert load_golden_set() == []


# ---------------------------------------------------------------------------
# HTTP seam: the review-preview response carries the warning, 200, no block
# ---------------------------------------------------------------------------

ACTORS = {"viewer": ("cb.chi-xem", "viewer")}
TOKENS = {role: f"tok-{role}" for role in ACTORS}


def _record() -> CanonicalRecord:
    return CanonicalRecord.model_validate(
        {
            "record_id": "DIEM_CHUAN:NGANH:2026",
            "record_type": "DIEM_CHUAN",
            "version": 1,
            "admission_year": 2026,
            "effective_from": "2026-01-01",
            "effective_to": "2026-12-31",
            "status": "draft",
            "authority_level": "husc_official",
            "source_document_id": "de-an-tuyen-sinh-2026",
            "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
            "source_page": 24,
            "text": "Diem chuan nam 2026 la 24.0 diem.",
        }
    )


class _FakePreviewService:
    def __init__(self, golden: GoldenRetrievalReport | None) -> None:
        self._golden = golden

    def diff_for(self, record_id, version=None):
        record = _record()
        return diff_records(None, record), record, (), record.status.value

    async def preview(self, record_id, query, version=None, top_k=5):
        record = _record()
        diff = diff_records(None, record)
        return ReviewPreviewResult(
            record_id=record_id,
            draft_version=record.version,
            baseline_version=None,
            high_impact_version=diff.high_impact_version,
            diff=diff,
            validation_errors=(),
            anomaly_warnings=(),
            preview_status="ok",
            answer="Diem chuan nam 2026 la 24.0 diem.",
            answer_status=None,
            model_answer=None,
            chunks=(),
            sources=(),
            golden_retrieval=self._golden,
        )


def _app(service, tmp_path) -> FastAPI:
    application = FastAPI()
    application.state.review_preview_service = service
    application.state.canonical_record_store = CanonicalRecordStore(tmp_path / "store")
    application.include_router(admin_preview_router)
    return application


def _headers() -> dict[str, str]:
    return {"X-Admin-Token": TOKENS["viewer"]}


PREVIEW_PATH = "/admin/records/DIEM_CHUAN:NGANH:2026/review-preview"

def _miss_report() -> GoldenRetrievalReport:
    """q001 finds its record, q002 lost it: the ticket's warning case."""
    retriever = _ScriptedRetriever(
        {
            "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026 là bao nhiêu?": [
                "HOC_PHI:nganh:7480201:2026:canon-nganh",
            ],
            "Mã ngành Kỹ thuật phần mềm là gì?": [
                "HOC_PHI:nganh:7480103:2026:canon-nganh",
            ],
        }
    )
    return run_golden_retrieval_check(GOLDEN, retriever, FakeEmbedder())


def test_preview_response_carrys_missing_question_warning_and_stays_200(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_IDENTITIES", __import__("json").dumps(
        [{"token": TOKENS["viewer"], "actor_id": "cb.chi-xem", "role": "viewer"}]
    ))
    report = _miss_report()
    client = TestClient(_app(_FakePreviewService(report), tmp_path))
    response = client.post(PREVIEW_PATH, json={"query": "Điểm chuẩn?"}, headers=_headers())
    assert response.status_code == 200, "cảnh báo không được chặn preview"
    body = response.json()
    assert body["preview_status"] == "ok"
    golden = body["golden_retrieval"]
    assert golden is not None
    assert golden["passed"] == 1 and golden["total"] == 2
    # (a) the warning names the exact question that lost its record:
    assert golden["failed"][0]["id"] == "q002"
    assert "q002" in golden["summary"]
    assert "không còn tìm thấy thông tin" in golden["summary"]


def test_preview_response_all_pass_shows_the_green_line(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_IDENTITIES", __import__("json").dumps(
        [{"token": TOKENS["viewer"], "actor_id": "cb.chi-xem", "role": "viewer"}]
    ))
    retriever = _ScriptedRetriever(
        {
            "Học phí tín chỉ ngành Công nghệ thông tin HUSC năm 2026 là bao nhiêu?": [
                "HOC_PHI:nganh:7480201:2026:canon-nganh",
            ],
            "Mã ngành Kỹ thuật phần mềm là gì?": [
                "NGANH:nganh:7480103:2026:canon-nganh",
            ],
        }
    )
    report = run_golden_retrieval_check(GOLDEN, retriever, FakeEmbedder())
    client = TestClient(_app(_FakePreviewService(report), tmp_path))
    response = client.post(PREVIEW_PATH, json={"query": "Điểm chuẩn?"}, headers=_headers())
    assert response.status_code == 200
    body = response.json()
    assert body["golden_retrieval"]["failed"] == []
    # (b) đủ → thông điệp x/y "vẫn tìm đúng thông tin":
    assert body["golden_retrieval"]["summary"] == (
        "2/2 câu hỏi mẫu vẫn tìm đúng thông tin"
    )


def test_preview_without_golden_set_serializes_none(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_IDENTITIES", __import__("json").dumps(
        [{"token": TOKENS["viewer"], "actor_id": "cb.chi-xem", "role": "viewer"}]
    ))
    client = TestClient(_app(_FakePreviewService(None), tmp_path))
    response = client.post(PREVIEW_PATH, json={"query": "Điểm chuẩn?"}, headers=_headers())
    assert response.status_code == 200
    assert response.json()["golden_retrieval"] is None


def test_unknown_record_still_404_even_with_golden_warning(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_IDENTITIES", __import__("json").dumps(
        [{"token": TOKENS["viewer"], "actor_id": "cb.chi-xem", "role": "viewer"}]
    ))

    class _RaisingService(_FakePreviewService):
        async def preview(self, record_id, query, version=None, top_k=5):
            raise RecordNotFoundError(f"no versions stored for {record_id!r}")

    client = TestClient(_app(_RaisingService(None), tmp_path))
    response = client.post(
        "/admin/records/khong-ton-tai/review-preview",
        json={"query": "?"},
        headers=_headers(),
    )
    assert response.status_code == 404


def test_real_preview_wiring_attaches_golden_report_over_the_temp_index(
    monkeypatch, tmp_path
):
    """Wiring proof (spec §02:87): the service's own temp index drives the check.

    One published record; a monkeypatched golden set expecting exactly that
    record is run through ``_run_real_path``'s real temp LanceDB + the real
    dense retrieve — the verdict must come from the index, not a fake.
    """
    from src.infrastructure.canonical_record_store import CanonicalRecordStore
    from src.services import golden_retrieval_check as gcheck
    from src.services.review_preview import ReviewPreviewService

    import src.main  # noqa: F401  (the real answer path must be loaded)

    store = CanonicalRecordStore(tmp_path / "canonical_store")
    rec = _record()
    store.write_record(rec)

    golden = [
        {
            "id": "q-w",
            "question": "Diem chuan?",
            "expected_record_ids": [rec.record_id],
        }
    ]
    monkeypatch.setattr(gcheck, "load_golden_set", lambda: golden)

    class _Emb:
        expected_dim = 8

        def encode_query(self, text):
            return [1.0] + [0.0] * 7

        def encode_batch(self, texts):
            return [[1.0] + [0.0] * 7 for _ in texts]

    class _Gen:
        """Enough of the generator copy the real path needs: the preview's
        synthesis probe + a canned provider answer (no LLM, offline)."""

        def __init__(self) -> None:
            self.retriever_probe = None

        async def generate_answer(self, **kwargs):
            return {
                "answer": "ok",
                "sources": [],
                "citations": [],
                "provider": "FakeModel(gen_ok)",
                "generation_failed": False,
                "model_answer": True,
            }

    service = ReviewPreviewService(
        store,
        embedder=_Emb(),
        embedding_dim=8,
        generator=_Gen(),
        llm=None,
    )
    import asyncio

    result = asyncio.run(service.preview(rec.record_id, "Diem chuan?", top_k=5))
    report = result.golden_retrieval
    assert report is not None, "preview must run the golden check on the temp index"
    assert report.passed == 1 and report.total == 1
    assert report.failed == ()
    assert report.summary == "1/1 câu hỏi mẫu vẫn tìm đúng thông tin"
