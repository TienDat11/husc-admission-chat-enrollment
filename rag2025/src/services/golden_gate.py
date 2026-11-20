"""Golden Gate - the blocking release-verification gate (U9 / spec §PL-6.7).

Run *every* golden question against the serving retriever of a release and
hard-enforce the invariants that the advisory T06 check only warned about:

* **26-case supersession matrix** - each of the 10 ``matrix_row`` values of
  §PL-6.7 must be present in the seed, and each case must pass
  (superseded old version must not appear in active top-k; new version must
  appear). Any count other than 26 cases, or any missing matrix row, fails *immediately*
  before any retrieval runs: ``verdict="fail"``,
  ``reason="incomplete_supersession_matrix"``.
* **Expected golden comparison** - match by ``coordinate_key``; Recall@5 and
  Recall@10 must not be below the baseline floor.
* **Citation precision** - by ``(document_id, page)``; anything below 100%
  fails.
* **Abstention** - out-of-scope questions that return an in-scope answer
  instead of abstaining fail.

A passing ``gating`` release moves to ``ready``; a failing one moves to
``gate_failed`` (which can never become ``active``).

Reuses :func:`load_golden_set` from ``src/services/golden_retrieval_check.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.services.golden_retrieval_check import (  # noqa: E402
    GoldenRetrievalReport,
    load_golden_set,
)

GOLDEN_SEED_PATH = Path(__file__).resolve().parents[2] / "tests" / "eval" / "fixtures" / "golden_seed.jsonl"

#: Supersession matrix rows (``matrix_row`` values) required by spec §PL-6.7.
#: Every row must be present in the seed and every case must pass.
SUPERSESSION_MATRIX_ROWS = (
    "1to1_same_id",
    "1to1_diff_id",
    "amends_dieu",
    "corrects",
    "repeals_expires",
    "split_1_to_n",
    "merge_n_to_1",
    "chain_a_b_c",
    "temporal_handoff",
    "coexist_revoke",
)
REQUIRED_MATRIX_COUNT = 26  # total supersession cases across the 10 rows


class GateError(Exception):
    """Raised when the golden gate cannot run at all."""


@dataclass(frozen=True)
class SupersessionCaseResult:
    question_id: str
    matrix_row: str
    relation: str
    old_keys: tuple[str, ...]
    new_keys: tuple[str, ...]
    new_in_top5: bool
    old_in_top5: bool
    ok: bool


@dataclass(frozen=True)
class GoldenGateReport:
    """Blocking gate verdict for one release."""

    release_id: str
    verdict: str  # "pass" | "fail"
    reason: str | None
    matrix_complete: bool
    matrix_passed: int
    matrix_total: int
    supersession: tuple[SupersessionCaseResult, ...]
    recall_at_5: float
    recall_at_10: float
    citation_precision: float
    abstention_ok: bool
    baseline: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "matrix_complete": self.matrix_complete,
            "matrix_passed": self.matrix_passed,
            "matrix_total": self.matrix_total,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "citation_precision": self.citation_precision,
            "abstention_ok": self.abstention_ok,
            "supersession": [
                {
                    "question_id": c.question_id,
                    "matrix_row": c.matrix_row,
                    "relation": c.relation,
                    "old_keys": list(c.old_keys),
                    "new_keys": list(c.new_keys),
                    "new_in_top5": c.new_in_top5,
                    "old_in_top5": c.old_in_top5,
                    "ok": c.ok,
                }
                for c in self.supersession
            ],
            "baseline": dict(self.baseline),
        }


@dataclass(frozen=True)
class _Chunk:
    record_id: str | None
    coordinate_keys: tuple[str, ...]
    pages: tuple[tuple[str, int], ...]


def _load_seed(path: Path | None = None) -> list[dict[str, Any]]:
    """Read the golden seed, preserving every field (``group``, ``matrix_row``,
    ``expected_coordinate_keys``, ``old/new_coordinate_keys`` ...).

    :func:`load_golden_set` is reused for the recall expected set: it trims each
    question to ``id``/``question``/``expected_record_ids``.
    """
    p = path or GOLDEN_SEED_PATH
    if not p.is_file():
        return []
    text = p.read_text(encoding="utf-8")
    try:
        artifact = json.loads(text)
    except json.JSONDecodeError:
        # JSONL: one question object per non-empty line.
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(artifact, dict) and "questions" in artifact:
        return [dict(q) for q in artifact["questions"]]
    if isinstance(artifact, list):
        return [dict(q) for q in artifact]
    return [dict(artifact)]


def _expected_record_ids(path: Path | None = None) -> list[dict[str, Any]]:
    """Canonical trimmed questions via :func:`load_golden_set` (reuse contract)."""
    return load_golden_set(path)


def _load_baseline(path: Path | None = None) -> dict[str, Any]:
    p = path or Path(__file__).resolve().parents[2] / "eval" / "baseline_A1.json"
    if not p.is_file():
        # Fail closed: a gate without its baseline floor cannot pass.
        raise FileNotFoundError(f"baseline file not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _matrix_rows(seed: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group supersession seed rows by their ``matrix_row`` value."""
    rows: dict[str, list[dict[str, Any]]] = {}
    for q in seed:
        if q.get("group") != "supersession":
            continue
        rows.setdefault(str(q.get("matrix_row") or ""), []).append(dict(q))
    return rows


def _matrix_complete(matrix_rows: Mapping[str, Sequence[Any]]) -> tuple[list[str], int, bool]:
    """Return ``(missing_rows, total_cases, complete)`` for the §PL-6.7 matrix."""
    missing = [row for row in SUPERSESSION_MATRIX_ROWS if row not in matrix_rows]
    total_cases = sum(len(v) for v in matrix_rows.values())
    return missing, total_cases, not missing and total_cases == REQUIRED_MATRIX_COUNT


def run_golden_gate(
    *,
    release_id: str,
    retriever: Any,
    embedder: Any,
    top_k: int = 10,
    seed_path: Path | None = None,
    baseline_path: Path | None = None,
) -> GoldenGateReport:
    """Run the blocking golden gate for ``release_id``.

    ``retriever`` must be duck-typed like :class:`LanceDBRetriever` and accept
    the same ``retrieve`` keyword arguments. ``embedder`` must expose
    ``encode_query``. Any retriever/embedder (real or fake) works - this is
    the seam the tests (and a PG-backed job) plug through.
    """
    seed = _load_seed(seed_path)
    baseline = _load_baseline(baseline_path)

    # ---- Supersession matrix: fail immediately if incomplete (no retrieval). ----
    matrix_rows = _matrix_rows(seed)
    missing, total_cases, matrix_complete = _matrix_complete(matrix_rows)

    if not matrix_complete:
        reason = "incomplete_supersession_matrix"
        return GoldenGateReport(
            release_id=release_id,
            verdict="fail",
            reason=reason,
            matrix_complete=False,
            matrix_passed=0,
            matrix_total=total_cases,
            supersession=(),
            recall_at_5=0.0,
            recall_at_10=0.0,
            citation_precision=0.0,
            abstention_ok=True,
            baseline=baseline,
        )

    # ---- Run every golden question through top-k retrieval. ----
    supersession_results: list[SupersessionCaseResult] = []
    in_scope_hits_5 = 0
    in_scope_hits_10 = 0
    in_scope_rows = 0
    citation_returned = 0
    citation_relevant = 0
    oos_rows = 0
    oos_abstained = 0

    for q in seed:
        group = str(q.get("group") or "")
        question = str(q.get("question") or "")
        qid = str(q.get("id") or "")

        top10_chunks = _top_chunks(retriever, embedder, question, top_k=10)
        top5_chunks = top10_chunks[:5]
        top5 = _record_keys(top5_chunks)
        top10 = _record_keys(top10_chunks)

        if group == "in_scope":
            expected = {str(k) for k in (q.get("expected_coordinate_keys") or [])}
            if not expected:
                continue
            expected_page_pairs = {
                (str(p["document_id"]), int(p["page"]))
                for p in (q.get("expected_pages") or [])
                if isinstance(p, Mapping)
                and p.get("document_id") is not None
                and p.get("page") is not None
            }
            in_scope_rows += 1
            if expected & top5:
                in_scope_hits_5 += 1
            if expected & top10:
                in_scope_hits_10 += 1
            # Citation precision over the top 5: a chunk with page metadata is
            # relevant only when a page matches; otherwise its record id or
            # coordinate keys must name an expected coordinate key.
            for ck in top5_chunks:
                if ck.record_id is None and not ck.coordinate_keys and not ck.pages:
                    continue
                citation_returned += 1
                if expected_page_pairs and ck.pages:
                    relevant = bool(set(ck.pages) & expected_page_pairs)
                else:
                    relevant = (ck.record_id is not None and ck.record_id in expected) or bool(
                        set(ck.coordinate_keys) & expected
                    )
                if relevant:
                    citation_relevant += 1

        elif group == "supersession":
            matrix_row = str(q.get("matrix_row") or "")
            relation = str(q.get("relation") or "")
            old_keys = tuple(str(k) for k in (q.get("old_coordinate_keys") or []))
            new_keys = tuple(str(k) for k in (q.get("new_coordinate_keys") or []))
            # new version must appear; old (superseded) version must not.
            new_in_top5 = bool(set(new_keys) & top5)
            old_in_top5 = bool(set(old_keys) & top5)
            ok = new_in_top5 and not old_in_top5
            supersession_results.append(
                SupersessionCaseResult(
                    question_id=qid,
                    matrix_row=matrix_row,
                    relation=relation,
                    old_keys=old_keys,
                    new_keys=new_keys,
                    new_in_top5=new_in_top5,
                    old_in_top5=old_in_top5,
                    ok=ok,
                )
            )

        elif group == "out_of_scope":
            expect_abstain = bool(q.get("expect_abstain") or False)
            if not expect_abstain:
                continue
            oos_rows += 1
            # Abstained iff nothing relevant came back in top-k.
            if not top5:
                oos_abstained += 1

    recall_at_5 = (in_scope_hits_5 / in_scope_rows) if in_scope_rows else 0.0
    recall_at_10 = (in_scope_hits_10 / in_scope_rows) if in_scope_rows else 0.0
    citation_precision = (citation_relevant / citation_returned) if citation_returned else 1.0
    abstention_ok = (oos_abstained / oos_rows) == 1.0 if oos_rows else True

    # ---- Verdict. ----
    matrix_passed = sum(1 for c in supersession_results if c.ok)
    fail_reason: str | None = None
    if matrix_passed != len(supersession_results):
        fail_reason = "supersession_case_failed"
    elif recall_at_5 < float(baseline.get("recall_at_5", 0.0)):
        fail_reason = "recall_at_5_below_baseline"
    elif recall_at_10 < float(baseline.get("recall_at_10", 0.0)):
        fail_reason = "recall_at_10_below_baseline"
    elif citation_precision < 1.0:
        fail_reason = "citation_precision_below_100"
    elif not abstention_ok:
        fail_reason = "abstention_failed"

    verdict = "fail" if fail_reason else "pass"
    return GoldenGateReport(
        release_id=release_id,
        verdict=verdict,
        reason=fail_reason,
        matrix_complete=True,
        matrix_passed=matrix_passed,
        matrix_total=total_cases,
        supersession=tuple(supersession_results),
        recall_at_5=recall_at_5,
        recall_at_10=recall_at_10,
        citation_precision=citation_precision,
        abstention_ok=abstention_ok,
        baseline=baseline,
    )


def _top_chunks(
    retriever: Any, embedder: Any, question: str, *, top_k: int = 10
) -> tuple[_Chunk, ...]:
    """Return the top-``top_k`` retrieved chunks as (record_id, keys, pages).

    One ``retrieve`` call per question: callers slice the result for top-5 and
    use the whole list for top-10.
    """
    qvec = embedder.encode_query(question)
    result = retriever.retrieve(query_vector=list(qvec), top_k=top_k, query=question)
    docs = result.documents if getattr(result, "is_success", True) else []
    chunks: list[_Chunk] = []
    for doc in docs:
        meta = doc.metadata or {}
        record_id = meta.get("record_id")
        keys = tuple(str(k) for k in (meta.get("coordinate_keys") or []))
        pages = tuple(
            (str(p["document_id"]), int(p["page"]))
            for p in (meta.get("pages") or [])
            if isinstance(p, Mapping)
            and p.get("document_id") is not None
            and p.get("page") is not None
        )
        chunks.append(_Chunk(record_id=record_id, coordinate_keys=keys, pages=pages))
    return tuple(chunks)


def _record_keys(chunks: Sequence[_Chunk]) -> set[str]:
    """Record ids and coordinate keys behind ``chunks``."""
    keys: set[str] = set()
    for ck in chunks:
        if ck.record_id:
            keys.add(str(ck.record_id))
        keys.update(ck.coordinate_keys)
    return keys


def _pg_url() -> str:
    import os

    url = os.environ.get(
        "DATABASE_URL", "postgresql+asyncpg://rag:rag@127.0.0.1:5432/husc"
    )
    # psycopg (sync) cannot parse the SQLAlchemy asyncpg driver prefix.
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def persist_to_pg(
    release_id: str,
    report: GoldenGateReport,
    database_url: str | None = None,
) -> None:
    """Persist ``golden_report`` and flip ``releases.status`` for ``release_id``.

    ``gating`` -> ``ready`` on pass, ``gate_failed`` on fail. No-op (silent)
    when the release row does not exist - callers that need a hard failure
    check ``release_exists`` first.
    """
    import psycopg

    from src.infrastructure.sql_record_store import _conninfo

    database_url = database_url or _pg_url()
    status = "ready" if report.verdict == "pass" else "gate_failed"
    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    with psycopg.connect(_conninfo(database_url), autocommit=True) as conn:
        with conn.cursor() as cur:
            # Only a release still being gated is written; the CASE moves it to ready/gate_failed.
            cur.execute(
                """
                UPDATE releases
                SET golden_report = %s::jsonb,
                    status = CASE WHEN status IN ('building', 'gating') THEN %s ELSE status END
                WHERE release_id = %s AND status IN ('building', 'gating')
                """,
                (payload, status, release_id),
            )


def release_exists(release_id: str, database_url: str) -> bool:
    """True when a ``releases`` row for ``release_id`` exists (any status)."""
    import psycopg

    from src.infrastructure.sql_record_store import _conninfo

    with psycopg.connect(_conninfo(database_url), autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM releases WHERE release_id = %s LIMIT 1", (release_id,))
            return cur.fetchone() is not None
