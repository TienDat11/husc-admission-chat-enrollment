"""TDD: the review-preview service — ticket 25.

The ticket's load-bearing claims, tested offline against the REAL machinery
(temp LanceDB + temp graph + the real ticket-06 chunker + the real
``src.main._synthesize_v2_answer``), with only the two network boundaries
faked — the embedder and the answer generator:

- the preview answer comes from the real /v2 synthesis path, not a private
  copy (identity + call-time resolution asserted);
- the draft is substituted as-if-published: the chunks the answer is built
  from carry the DRAFT's text, never the superseded published one;
- a record that could not be published never yields an answer
  (``blocked_validation`` suppresses generation entirely — the exploding
  embedder proves the index was never even built);
- a blocked provider reports ``insufficient_evidence_provider`` and STILL
  returns chunks/sources, so the reviewer sees the grounding was unverified
  (never a faked "verified" answer);
- everything the preview writes lands in its temporary directory — the
  production table / materializer / chunk-index / NER-cache paths are
  asserted untouched.

``router_result`` is the service's documented test seam: it pins the route to
PADDED_RAG, which the pipeline answers from vector docs alone (no graph
retrieve, no router LLM), keeping the whole file offline and deterministic.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import lancedb
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.main  # noqa: E402,F401  (the real answer path must be loaded)
from config.settings import settings  # noqa: E402
from src.domain.canonical.record import CanonicalRecord  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.services import ner_service as ner_service_module  # noqa: E402
from src.services import review_preview as review_preview_module  # noqa: E402
from src.services.query_router import QueryRoute, RouterResult  # noqa: E402
from src.services.review_preview import (  # noqa: E402
    PreviewPathUnavailable,
    ReviewPreviewService,
    validate_for_publication,
)

RECORD_ID = "DIEM_CHUAN:NGANH:7480201:2026"
TEXT_V1 = "Diem chuan nganh Cong nghe thong tin nam 2026 la 22.5 diem."
TEXT_V2 = "Diem chuan nganh Cong nghe thong tin nam 2026 la 24.0 diem."
QUERY = "Điểm chuẩn ngành Công nghệ thông tin năm 2026 là bao nhiêu?"


def _record(version: int, status: str, text: str, **overrides) -> CanonicalRecord:
    payload = {
        "record_id": RECORD_ID,
        "record_type": "DIEM_CHUAN",
        "version": version,
        "admission_year": 2026,
        "effective_from": date(2026, 1, 1),
        "effective_to": date(2026, 12, 31),
        "status": status,
        "authority_level": "husc_official",
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 24,
        "published_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "text": text,
    }
    payload.update(overrides)
    return CanonicalRecord.model_validate(payload)


class FakeEmbedder:
    """Deterministic offline stand-in for EmbeddingService (dim 8)."""

    def __init__(self, dim: int = 8, query_vector=None):
        self.dim = dim
        self.expected_dim = dim
        default_q = [1.0] + [0.0] * (dim - 1)
        self._query_vector = default_q if query_vector is None else query_vector

    def encode_query(self, text: str) -> list[float]:
        return list(self._query_vector)

    def encode_batch(self, texts) -> list[list[float]]:
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


class ExplodingEmbedder(FakeEmbedder):
    """Proves a blocked preview never reaches the index builder."""

    def encode_query(self, text):
        raise AssertionError("blocked preview must not build the temp index")

    def encode_batch(self, texts):
        raise AssertionError("blocked preview must not build the temp index")


class FakeGenerator:
    """Stand-in for LLMGenerator: records the synthesis call, returns the
    same payload keys the real generator publishes."""

    def __init__(self, *, answer, provider, generation_failed=False):
        self.answer = answer
        self.provider = provider
        self.generation_failed = generation_failed
        self.calls: list[dict] = []

    async def generate_answer(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "answer": self.answer,
            "sources": [],
            "citations": [],
            "provider": self.provider,
            "generation_failed": self.generation_failed,
            "model_answer": not self.generation_failed,
        }


def _padded_router_result(query: str = QUERY) -> RouterResult:
    return RouterResult(
        original_query=query,
        step_back_query="",
        hypothetical_doc="",
        hyde_variants=[],
        route=QueryRoute.PADDED_RAG,
        complexity=1,
        intent="diem_chuan",
        reasoning="test seam",
    )


@pytest.fixture(autouse=True)
def _pin_admission_year(monkeypatch):
    # The temporal post-filter resolves the year per call; pin it so the
    # 2026 fixture rows are current-year regardless of the wall clock.
    monkeypatch.setenv("CURRENT_ADMISSION_YEAR", "2026")


def _store(tmp_path: Path, latest_status: str = "draft", **latest_overrides):
    s = CanonicalRecordStore(tmp_path / "canonical_store")
    s.write_record(_record(1, "published", TEXT_V1))
    s.edit_record(
        RECORD_ID, status=latest_status, text=TEXT_V2, **latest_overrides
    )
    return s


@pytest.fixture
def store(tmp_path: Path) -> CanonicalRecordStore:
    return _store(tmp_path)


@pytest.fixture
def writers(monkeypatch) -> dict:
    """Spy the writers the temp context uses, recording every path handed to
    each. Delegates to the real implementations — the preview still runs."""
    recorded: dict[str, list[str]] = {
        "lancedb": [],
        "materialize": [],
        "chunk_index": [],
        "ner_cache": [],
    }

    real_connect = lancedb.connect

    def connect_spy(uri, *args, **kwargs):
        recorded["lancedb"].append(str(uri))
        return real_connect(uri, *args, **kwargs)

    monkeypatch.setattr(lancedb, "connect", connect_spy)

    real_materializer = review_preview_module.PublishedMaterializer

    class MaterializerSpy(real_materializer):
        def materialize(self, release_id, output_dir, *args, **kwargs):
            recorded["materialize"].append(str(output_dir))
            return super().materialize(release_id, output_dir, *args, **kwargs)

    monkeypatch.setattr(
        review_preview_module, "PublishedMaterializer", MaterializerSpy
    )

    real_build = review_preview_module.build_chunk_index

    def build_spy(records, out_dir, *args, **kwargs):
        recorded["chunk_index"].append(str(out_dir))
        return real_build(records, out_dir, *args, **kwargs)

    monkeypatch.setattr(review_preview_module, "build_chunk_index", build_spy)

    real_ner = ner_service_module.NERService

    class NerSpy(real_ner):
        def __init__(self, *args, **kwargs):
            cache = kwargs.get("cache_path")
            if cache is not None:
                recorded["ner_cache"].append(str(cache))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        review_preview_module, "NERService", NerSpy, raising=False
    )
    # The service imports NERService inside _run_real_path from its module.
    monkeypatch.setattr(ner_service_module, "NERService", NerSpy)
    return recorded


def _service(store, embedder, generator) -> ReviewPreviewService:
    return ReviewPreviewService(
        store,
        embedder=embedder,
        embedding_dim=embedder.expected_dim,
        generator=generator,
        llm=None,
    )


def _preview(service, **kwargs):
    kwargs.setdefault("router_result", _padded_router_result())
    return asyncio.run(service.preview(RECORD_ID, QUERY, **kwargs))


# ---------------------------------------------------------------------------
# criterion 6: the real answer path, not a private copy
# ---------------------------------------------------------------------------

def test_synthesize_fn_is_the_real_main_helper(monkeypatch):
    # Some probes load src.main a second time under another module name (the
    # rate-limit guard uses spec_from_file_location). The service resolves
    # ("main", "src.main", "__main__") in order, so a stale "main" alias would
    # out-rank the canonical copy — drop it and assert against the copy the
    # test suite itself imports.
    monkeypatch.delitem(sys.modules, "main", raising=False)
    assert (
        ReviewPreviewService._synthesize_fn() is src.main._synthesize_v2_answer
    ), "preview must reuse /v2's synthesis, not a copy that can drift"


def test_synthesize_fn_fails_loud_without_the_real_path(monkeypatch):
    # The lookup walks sys.modules per call: if the host module disappears,
    # the preview must fail LOUD, never fall back to a private imitation.
    monkeypatch.delitem(sys.modules, "main", raising=False)
    monkeypatch.delitem(sys.modules, "src.main", raising=False)
    monkeypatch.delattr(sys.modules["__main__"], "_synthesize_v2_answer", raising=False)
    with pytest.raises(PreviewPathUnavailable):
        ReviewPreviewService._synthesize_fn()


def test_preview_synthesizes_through_main_helper(store, monkeypatch):
    # Call-time resolution, asserted behaviorally: patch the attribute on the
    # loaded module and the preview must pick the replacement up.
    calls: list[tuple] = []
    real = src.main._synthesize_v2_answer

    async def spy(rag_result, query, generator, history=None):
        calls.append((rag_result, query))
        return await real(rag_result, query, generator, history=history)

    monkeypatch.setattr(src.main, "_synthesize_v2_answer", spy)
    generator = FakeGenerator(answer="24.0 diem.", provider="FakeModel(gen_ok)")
    result = _preview(_service(store, FakeEmbedder(), generator))
    assert calls, "preview ran a synthesis that is not main's helper"
    assert result.preview_status == "ok"


# ---------------------------------------------------------------------------
# end-to-end preview (offline): ok / provider-blocked / no-evidence
# ---------------------------------------------------------------------------


def test_preview_ok_answers_from_the_draft_via_the_real_path(store, writers):
    generator = FakeGenerator(
        answer="Điểm chuẩn năm 2026 là 24.0 điểm.", provider="FakeModel(gen_ok)"
    )
    result = _preview(_service(store, FakeEmbedder(), generator))

    assert result.preview_status == "ok"
    assert result.answer == "Điểm chuẩn năm 2026 là 24.0 điểm."
    assert result.model_answer is True
    assert result.baseline_version == 1 and result.draft_version == 2
    # The draft was substituted as-if-published: the grounding the generator
    # saw carries the DRAFT's number, never the superseded published one.
    assert generator.calls, "the real synthesis must have called the generator"
    chunk_texts = [c["text"] for c in generator.calls[0]["chunks"]]
    assert any("24.0" in t for t in chunk_texts)
    assert not any("22.5" in t for t in chunk_texts)
    # The returned chunks/sources are the temp index's real rows.
    assert result.chunks and "24.0" in result.chunks[0]["text"]
    assert result.sources[0]["source_document_id"] == "de-an-tuyen-sinh-2026"
    # The diff rides along: same diff the review-diff endpoint shows. The
    # published -> draft lifecycle move is a real field change, but only
    # `text` reaches the student, so only `text` is high-impact.
    assert [c.field for c in result.diff.changes] == ["status", "text"]
    assert [c.field for c in result.diff.high_impact_changes] == ["text"]


def test_preview_blocked_provider_reports_provider_status_and_keeps_grounding(
    store, writers
):
    # The ticket's literal rule: a blocked provider must NOT pretend the
    # answer was verified — answer is None but the chunks/sources the
    # reviewer must inspect are still returned.
    generator = FakeGenerator(
        answer="NEVER SHOWN", provider="Fallback(NoProvider)", generation_failed=True
    )
    result = _preview(_service(store, FakeEmbedder(), generator))

    assert result.preview_status == "insufficient_evidence_provider"
    assert result.answer is None
    assert result.chunks, "provider-blocked preview must still show the grounding"
    assert result.sources
    assert generator.calls, "the real path must still have attempted generation"


def test_preview_without_retrieved_docs_is_insufficient_evidence(store, writers):
    # Empty query vector -> the retriever's own INVALID_PAYLOAD guard ->
    # zero docs -> no answer may be presented as grounded.
    generator = FakeGenerator(answer="X", provider="FakeModel(gen_ok)")
    embedder = FakeEmbedder(query_vector=[])
    result = _preview(_service(store, embedder, generator))

    assert result.preview_status == "insufficient_evidence"
    assert result.answer is None
    assert result.chunks == ()
    assert result.sources == ()


# ---------------------------------------------------------------------------
# criterion 5: validation errors block submission, answer suppressed
# ---------------------------------------------------------------------------


def test_preview_blocked_validation_suppresses_answer_and_names_field(tmp_path):
    store = _store(tmp_path, source_document_id=None, source_url=None)
    service = _service(store, ExplodingEmbedder(), FakeGenerator(answer="x", provider="p"))
    result = _preview(service)

    assert result.preview_status == "blocked_validation"
    assert result.answer is None
    fields = {v.field for v in result.validation_errors}
    assert "source_document_id" in fields
    # The diff still returns — the reviewer sees WHAT changed alongside
    # what must be fixed.
    assert [c.field for c in result.diff.changes] == [
        "source_document_id",
        "source_url",
        "status",
        "text",
    ]
    # ExplodingEmbedder would have raised had the index been built.


def test_validate_for_publication_passes_a_publishable_draft(store):
    draft = store.get_record(RECORD_ID).latest
    assert validate_for_publication(draft) == ()


def test_preview_refuses_non_previewable_status(tmp_path):
    # A published version IS the live answer; previewing it (or an archived
    # one) must be blocked with a field error, not a bogus preview.
    store = _store(tmp_path, latest_status="published")
    service = _service(store, ExplodingEmbedder(), FakeGenerator(answer="x", provider="p"))
    result = _preview(service)

    assert result.preview_status == "blocked_validation"
    assert any(v.field == "status" for v in result.validation_errors)
    assert result.answer is None


# ---------------------------------------------------------------------------
# diff_for: the review-diff endpoint's data, without building any index
# ---------------------------------------------------------------------------


def test_diff_for_shows_draft_vs_published_baseline(store):
    diff, draft, issues, status = _service(
        store, ExplodingEmbedder(), FakeGenerator(answer="x", provider="p")
    ).diff_for(RECORD_ID)
    assert status == "draft"
    assert draft.version == 2
    assert diff.baseline_version == 1
    assert [c.field for c in diff.changes] == ["status", "text"]
    assert [c.field for c in diff.high_impact_changes] == ["text"]
    assert diff.has_high_impact is True
    assert issues == ()


def test_diff_for_selects_a_specific_version(store):
    diff, draft, _issues, _status = _service(
        store, ExplodingEmbedder(), FakeGenerator(answer="x", provider="p")
    ).diff_for(RECORD_ID, version=1)
    assert draft.version == 1
    # v1 IS the published baseline: comparing it against itself is no change.
    assert diff.changes == ()


# ---------------------------------------------------------------------------
# the ticket's core safety property: nothing lands in production
# ---------------------------------------------------------------------------


def _dir_listing(root: Path) -> dict:
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_preview_writes_only_inside_its_temp_directory(store, writers):
    production = {
        "lancedb": Path(settings.LANCEDB_URI),
        "ner_cache": Path(ner_service_module.DEFAULT_NER_CACHE_PATH),
    }
    before = {
        name: (
            _dir_listing(path) if path.is_dir()
            else (path.stat().st_mtime_ns if path.exists() else None)
        )
        for name, path in production.items()
    }

    generator = FakeGenerator(answer="24.0.", provider="FakeModel(gen_ok)")
    result = _preview(_service(store, FakeEmbedder(), generator))
    assert result.preview_status == "ok"

    temp_root = str(Path(tempfile.gettempdir()))
    # Every path handed to a writer was inside the preview's own temp dir.
    for kind in ("lancedb", "materialize", "chunk_index", "ner_cache"):
        assert writers[kind], f"preview never exercised the {kind} writer"
        for path in writers[kind]:
            assert path.startswith(temp_root), (
                f"{kind} writer was pointed OUTSIDE the temp dir: {path}"
            )
    # And the production paths are byte-for-byte untouched.
    after = {
        name: (
            _dir_listing(path) if path.is_dir()
            else (path.stat().st_mtime_ns if path.exists() else None)
        )
        for name, path in production.items()
    }
    assert after == before, "preview modified a production data path"


# ---------------------------------------------------------------------------
# T19: anomaly warnings reach the preview response without blocking publication
# ---------------------------------------------------------------------------


def test_preview_anomaly_warnings_appear_in_response(tmp_path, writers):
    # Store a draft with a zero tuition value; the preview must surface it
    # under ``anomaly_warnings`` and NOT mix it into ``validation_errors``.
    store = CanonicalRecordStore(tmp_path / "canonical_store")
    record_v1 = _record(1, "published", TEXT_V1)
    store.write_record(record_v1)
    store.edit_record(
        RECORD_ID,
        status="draft",
        text=TEXT_V2,
        values={"hoc_phi_tin_chi_vnd": 0, "ten_nganh": "CNTT"},
    )
    generator = FakeGenerator(answer="x", provider="FakeModel(gen_ok)")
    result = _preview(_service(store, FakeEmbedder(), generator))

    # The warning is present and names the offending field.
    warnings = result.anomaly_warnings
    assert any(
        a.field == "hoc_phi_tin_chi_vnd" and a.rule == "zero-implausible"
        for a in warnings
    ), f"expected zero-implausible warning, got: {warnings!r}"
    # The warning is NOT in validation_errors.
    assert not any(v.field == "hoc_phi_tin_chi_vnd" for v in result.validation_errors)
    # The preview still produced an answer (warnings don't block).
    assert result.preview_status == "ok"
    # to_dict exposes the key for the admin API.
    d = result.to_dict()
    assert "anomaly_warnings" in d
    assert any(
        w["field"] == "hoc_phi_tin_chi_vnd" and w["rule"] == "zero-implausible"
        for w in d["anomaly_warnings"]
    )


def test_preview_no_anomaly_warnings_for_clean_record(store, writers):
    # A record with no zero / negative values yields an empty tuple.
    generator = FakeGenerator(answer="x", provider="FakeModel(gen_ok)")
    result = _preview(_service(store, FakeEmbedder(), generator))
    assert result.anomaly_warnings == ()


# ---------------------------------------------------------------------------
# T05: a draft carrying pattern PII never reaches the index or the provider
# ---------------------------------------------------------------------------


def test_preview_blocked_pii_suppresses_synthesis(tmp_path):
    # CCCD: 12 digits -> the citizen_id pattern fires on the DRAFT text.
    pii_text = (
        "Danh sach trung tuyen bo sung, lien he phong dao tao. "
        "Can cuoc cong dan 001234567890 de doi chieu ho so nhap hoc."
    )
    store = CanonicalRecordStore(tmp_path / "canonical_store")
    store.write_record(_record(1, "published", TEXT_V1))
    store.edit_record(RECORD_ID, status="draft", text=pii_text)
    generator = FakeGenerator(answer="NEVER SHOWN", provider="FakeModel(gen_ok)")
    result = _preview(_service(store, ExplodingEmbedder(), generator))

    assert result.preview_status == "blocked_pii"
    assert result.answer is None
    assert result.chunks == ()
    assert result.sources == ()
    fields = {v.field for v in result.validation_errors}
    assert fields == {"draft_text"}
    assert any("citizen_id" in v.message for v in result.validation_errors)
    assert generator.calls == [], "blocked preview must not call the provider"
    # ExplodingEmbedder would have raised had the temp index been built.


def test_preview_blocked_pii_in_structured_values(tmp_path):
    """Defect 2: PII buried in the draft's structured ``values`` payload —
    not in ``text`` — must still block the preview. The gate scans every
    unit the chunk builder will index, so a phone number in a field value
    reaches the external generator no differently than one in the body
    text."""
    pii_value = "0912345678"
    clean_text = "Diem chuan nganh Cong nghe thong tin nam 2026 la 24.0 diem."
    store = CanonicalRecordStore(tmp_path / "canonical_store")
    store.write_record(_record(1, "published", TEXT_V1))
    store.edit_record(
        RECORD_ID,
        status="draft",
        text=clean_text,
        values={"ten_nganh": "CNTT lien he 0912345678"},
    )
    generator = FakeGenerator(answer="NEVER SHOWN", provider="FakeModel(gen_ok)")
    result = _preview(_service(store, ExplodingEmbedder(), generator))

    assert result.preview_status == "blocked_pii"
    assert result.answer is None
    assert result.chunks == ()
    assert result.sources == ()
    fields = {v.field for v in result.validation_errors}
    assert fields == {"draft_text"}
    assert any("phone" in v.message for v in result.validation_errors), (
        f"expected a phone flag, got: {[v.message for v in result.validation_errors]}"
    )
    assert pii_value not in str(result.validation_errors)
    assert generator.calls == [], "blocked preview must not call the provider"
    # ExplodingEmbedder would have raised had the temp index been built.
