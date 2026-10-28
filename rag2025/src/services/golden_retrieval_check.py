"""T06 — lớp (d) rút gọn: kiểm câu hỏi mẫu mức truy xuất trước xuất bản.

Spec §02:87 — MVP gate là mức truy xuất, KHÔNG gọi LLM: mỗi câu vàng có
``expected_record_ids`` → kiểm record có trong top-5 trên index tạm; cảnh
báo, không chặn publish.

Golden set: artifact ``data/eval/husc86q_golden_records.json`` (86q +
heldout, cột ``expected_record_ids`` do ``scripts/build_golden_records.py``
sinh). Chỉ các câu có kỳ vọng không rỗng được kiểm — câu không có kỳ vọng
không thể kiểm top-5 mà không bịa.

Spec §05:61 UX: "8/8 câu hỏi mẫu vẫn tìm đúng thông tin" /
"⚠ N câu hỏi mẫu không còn tìm thấy thông tin: …".
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from src.domain.canonical.record import CanonicalRecord
from src.services.lancedb_retrieval import LanceDBRetriever

_RAG2025_ROOT = Path(__file__).resolve().parents[2]

#: Golden set + heldout questions that carry an expected record set.
GOLDEN_RECORDS_FILENAME = "husc86q_golden_records.json"


def golden_set_path() -> Path:
    """The golden records artifact the check runs against."""
    return _RAG2025_ROOT / "data" / "eval" / GOLDEN_RECORDS_FILENAME


def load_golden_set(path: Optional[Path] = None) -> list[dict]:
    """Golden questions with non-empty ``expected_record_ids``.

    The artifact (T00 build) holds one entry per question: ``id``,
    ``question`` and ``expected_record_ids``. Entries without an expected
    set are skipped — a question nobody derived an expectation for cannot
    be checked against top-5 without inventing one.
    """
    p = path or golden_set_path()
    if not p.is_file():
        return []
    artifact = json.loads(p.read_text(encoding="utf-8"))
    questions = artifact.get("questions", artifact if isinstance(artifact, list) else [])
    out: list[dict] = []
    for q in questions:
        expected = [str(r) for r in (q.get("expected_record_ids") or []) if r]
        if not expected:
            continue
        out.append(
            {
                "id": str(q.get("id") or ""),
                "question": str(q.get("question") or ""),
                "expected_record_ids": expected,
            }
        )
    return out


@dataclass(frozen=True)
class GoldenRetrievalReport:
    """T06 layer-(d) verdict: per-question top-5 retrieval check, advisory."""

    passed: int
    total: int
    failed: tuple[dict[str, str], ...]  # {"id","question","expected_record_ids"}

    @property
    def summary(self) -> str:
        """Spec §05:61 user-facing line, Vietnamese, plain words."""
        if self.failed:
            count = len(self.failed)
            head = f"⚠ {count} câu hỏi mẫu không còn tìm thấy thông tin:"
            items = " ".join(
                f"[{item['id']}]"
                if item.get("id")
                else f"({item.get('question', '')[:40]})"
                for item in self.failed
            )
            return f"{head} {items}"
        return f"{self.passed}/{self.total} câu hỏi mẫu vẫn tìm đúng thông tin"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total": self.total,
            "failed": [dict(item) for item in self.failed],
            "summary": self.summary,
        }


def _top5_record_ids(retriever: LanceDBRetriever, embedder: Any, question: str) -> set[str]:
    """Dense top-5 over the (temporary) index: record ids behind the docs.

    No LLM: one query vector + the retriever's real dense path, the same
    arm ``POST /v2/query`` grounds its answer on.
    """
    qvec = embedder.encode_query(question)
    result = retriever.retrieve(query_vector=list(qvec), top_k=5, query=question)
    docs = result.documents if result.is_success else []
    ids: set[str] = set()
    for doc in docs:
        record_id = (doc.metadata or {}).get("record_id")
        if record_id:
            ids.add(str(record_id))
    return ids


def run_golden_retrieval_check(
    golden: Sequence[Mapping[str, Any]],
    retriever: LanceDBRetriever,
    embedder: Any,
) -> GoldenRetrievalReport:
    """Run every golden question through top-5; report misses, never block."""
    failed: list[dict[str, str]] = []
    passed = 0
    for q in golden:
        expected = {str(r) for r in q.get("expected_record_ids") or []}
        if not expected:
            continue
        found = _top5_record_ids(retriever, embedder, str(q.get("question") or ""))
        if expected & found:
            passed += 1
        else:
            failed.append(
                {
                    "id": str(q.get("id") or ""),
                    "question": str(q.get("question") or ""),
                    "expected_record_ids": sorted(expected),
                }
            )
    return GoldenRetrievalReport(passed=passed, total=passed + len(failed), failed=tuple(failed))
