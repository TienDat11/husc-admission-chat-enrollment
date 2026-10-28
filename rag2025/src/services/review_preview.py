"""Review preview: see the diff and the student answer *before* submitting.

Ticket 25 (.scratch/husc-rag-admin-upgrade/issues/25-admin-review-diff-preview.md).

A reviewer editing a canonical record needs two things before the record goes
to ``review``:

1. The **diff** against the currently published version, with the high-impact
   fields flagged (computed server-side in ``src.domain.canonical.diff``, so
   API consumers and the audit trail see the same verdict the UI does).
2. The **answer a student would receive** if this draft were published —
   produced by the real answer path, not a simulation.

Why a temporary index
---------------------
The real answer path resolves materialized text through the *published*
LanceDB corpus. A draft is not published, so its new text is not in that
index; querying production would show the reviewer the OLD answer. The only
honest preview builds a throwaway index (LanceDB table + corpus graph) from
the published baseline with the draft substituted as if published, and runs
the same retrieval/synthesis/generation code on top of it. Nothing here ever
writes to the production LanceDB table, the production GraphML, or any cache
— including the NER cache, whose default path is CWD-relative, so the
preview always passes a temp ``cache_path``.

Real-path guarantee
-------------------
- The synthesis leg is literally ``src.main._synthesize_v2_answer`` — the
  same function ``POST /v2/query`` calls. It is resolved from the module
  *already imported in this process* (``sys.modules``), never re-imported by
  name: a fresh ``import src.main`` would build a SECOND app instance whose
  lifespan never ran, and the identity assertion would compare against a
  copy (uvicorn also runs the file as ``main``, see the warning at the top
  of main.py). If no main module is loaded, the preview says so loudly.
- The generation leg is ``src.services.llm_generator.LLMGenerator`` — the
  same class production instantiates (default: the app's singleton).
- Retrieval uses the real ``LanceDBRetriever`` / ``UnifiedRAGPipeline``
  classes, bound to the temporary context instead of the module globals.
  Module globals are NEVER swapped: a request-time swap would let concurrent
  student traffic read the temp index at await points.

Embedding cost
--------------
Baseline chunks whose ``content_hash`` already has a vector in the serving
``vectors.jsonl`` reuse it (the rule ``ReindexJob._vectors_for`` uses); only
the draft's own chunks — and any baseline chunk whose text changed since the
last ingest — are embedded. The embed + index build is CPU/disk work and is
kept off the event loop via ``asyncio.to_thread``.

Known fidelity limits (stated, not silent)
------------------------------------------
- ``USE_HYBRID_RETRIEVAL`` (default False): when an operator enables the
  hybrid arm, production /v2 fuses BM25 and this preview (dense-only) can
  rank differently. Wiring the hybrid service over the temp index is a
  main.py seam (ticket 35 owns parameterizing it).
- The season-gap guard inside ``LLMGenerator`` probes the PRODUCTION
  retriever for "does year Y have data". A draft for a year absent from
  production can therefore preview as an abstention even though the temp
  index has the fact. That is faithful to what the generator would do on
  the day the record publishes into an un-ingested year; the reviewer sees
  the chunks/sources regardless.
- Router auto-answers (CONTACT_BLOCK etc.) are surfaced verbatim with
  ``preview_status="auto_answer"`` — the same "never swallow it" fix
  ``_synthesize_v2_answer`` carries.

Provider blocked, not faked
---------------------------
When the LLM provider is unavailable the real path degrades to
``generation_blocked`` (the state ``manifest_303q.json`` records 298 times).
The preview must not pretend an answer was grounded: per ticket 44 the
explicit ``generation_failed`` flag is the public signal (the
``Fallback(...)`` provider tag is only a safety net, so the flag OR
``is_generation_failure(provider)`` decides). The preview then reports
``preview_status = insufficient_evidence_provider``, answer ``None``, and
**still returns the retrieved chunks and sources**, so the reviewer can
inspect the grounding the answer would have had.
"""
from __future__ import annotations

import asyncio
import copy
import importlib
import importlib.util
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from pydantic import ValidationError

from src.domain.canonical.anomaly_rules import (
    Anomaly,
    find_anomalies,
)
from src.domain.canonical.diff import RecordDiff, diff_records
from src.domain.canonical.high_impact import HIGH_IMPACT_VERSION
from src.domain.canonical.record import CanonicalRecord, RecordStatus
from src.domain.graph import KnowledgeGraph
from src.infrastructure.canonical_record_store import CanonicalRecordStore
from src.services.canonical_chunk_builder import build_chunk_index
from src.services.lancedb_retrieval import LanceDBRetriever, LanceDBRetrieverConfig
from src.services.published_materializer import PublishedMaterializer

# Corpus graph relation used by the deterministic reindex path. Imported from
# its owner rather than restated, so the preview graph cannot drift from the
# real one.
from src.services.reindex_job import RELATION_HAS_CHUNK

# Ticket 44's contract: the explicit flag is the public signal, the provider
# prefix is the derived safety net. Import the predicate, never re-spell it.
from src.services.llm_generator import is_generation_failure
# PII gate for drafts: blocks preview before any index/build/provider call.
from src.services.pii_flags import blocking_pii_flags
from src.domain.canonical.semantic_units import split_field_units

# Statuses a reviewer may legitimately preview: work in the pipeline, not
# terminal states. ``published`` needs no preview (it IS the live answer);
# ``archived`` cannot be submitted for review.
PREVIEWABLE_STATUSES = frozenset(
    {RecordStatus.DRAFT, RecordStatus.REVIEW, RecordStatus.APPROVED}
)

_RAG2025_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ValidationIssue:
    """One field-level error blocking publication (criterion 5).

    ``field`` is the JSON field path the admin form marks red; ``message`` is
    the validator's own text, never a paraphrase.
    """

    field: str
    message: str

    def to_dict(self) -> dict:
        return {"field": self.field, "message": self.message}


def _field_for_message(msg: str) -> str:
    """Map a model-level ValueError to the field the admin must fix.

    The record's domain invariants raise from model validators with no field
    location, but each message names its field verbatim (see
    ``CanonicalRecord``'s published-state and effective-window validators).
    Matching on the message text is the only way to give the form a field to
    highlight; anything else falls back to the record as a whole.
    """
    lowered = msg.lower()
    if "source_document_id" in lowered:
        return "source_document_id"
    if "effective_from" in lowered or "effective_to" in lowered:
        return "effective_from"
    return ""


def validate_for_publication(record: CanonicalRecord) -> tuple[ValidationIssue, ...]:
    """Errors that would block ``publish`` of this version, per field.

    Publication requires a *published-state* record: source provenance and a
    sane effective window are enforced by the model's own validators, so the
    check is "re-validate a copy with status forced to published" — the
    record's single source of truth, not a second ruleset that could drift.
    ``model_copy`` deliberately does NOT run validators, which is why the copy
    round-trips through ``model_dump(mode="json")`` / ``model_validate``.
    """
    payload = record.model_dump(mode="json")
    payload["status"] = RecordStatus.PUBLISHED.value
    try:
        CanonicalRecord.model_validate(payload)
    except ValidationError as exc:
        issues: list[ValidationIssue] = []
        for err in exc.errors():
            loc = tuple(str(p) for p in err["loc"])
            field = ".".join(loc) if loc else _field_for_message(err["msg"])
            issues.append(ValidationIssue(field=field, message=err["msg"]))
        return tuple(issues)
    except ValueError as exc:  # model-level validators raise ValueError
        msg = str(exc)
        return (ValidationIssue(field=_field_for_message(msg), message=msg),)
    return ()


def published_version(store: CanonicalRecordStore, record_id: str) -> Optional[CanonicalRecord]:
    """The version currently published, or None if the record never was."""
    for version in reversed(store.versions(record_id)):
        record = store.get_version(record_id, version)
        if record.status is RecordStatus.PUBLISHED:
            return record
    return None


def _as_published(draft: CanonicalRecord) -> CanonicalRecord:
    """The draft as the indexer must see it: status forced to published.

    ``build_chunk_index`` refuses non-published records (ticket 06's
    admission gate) and ``canonical_corpus_rows`` re-checks the stamp — both
    gates stay untouched; the preview passes a *copy* that already cleared
    ``validate_for_publication``, so the forced status is one the model
    itself accepts.
    """
    return CanonicalRecord.model_validate(
        {**draft.model_dump(mode="json"), "status": RecordStatus.PUBLISHED.value}
    )


@dataclass(frozen=True)
class ReviewPreviewResult:
    """Everything the review screen renders: diff, validation, preview answer."""

    record_id: str
    draft_version: int
    baseline_version: Optional[int]
    high_impact_version: str
    diff: RecordDiff
    validation_errors: tuple[ValidationIssue, ...]
    anomaly_warnings: tuple[Anomaly, ...]
    # ok | auto_answer | blocked_validation | insufficient_evidence
    # | insufficient_evidence_provider | blocked_pii
    preview_status: str
    answer: Optional[str]
    answer_status: Optional[str]  # provider verdict verbatim, when generation ran
    model_answer: Optional[bool]  # a model wrote it (False: router-canned/fallback)
    chunks: tuple[dict, ...]
    sources: tuple[dict, ...]
    # T06 (layer d, spec §02:87): golden-question top-5 retrieval check over
    # the temp index — advisory, never blocks. None when the set is
    # unavailable (no golden artifact), a report otherwise.
    golden_retrieval: Optional[Any] = None

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "draft_version": self.draft_version,
            "baseline_version": self.baseline_version,
            "high_impact_version": self.high_impact_version,
            "diff": self.diff.to_dict(),
            "validation_errors": [v.to_dict() for v in self.validation_errors],
            "anomaly_warnings": [a.to_dict() for a in self.anomaly_warnings],
            "preview_status": self.preview_status,
            "answer": self.answer,
            "answer_status": self.answer_status,
            "model_answer": self.model_answer,
            "chunks": [dict(c) for c in self.chunks],
            "sources": [dict(s) for s in self.sources],
            "golden_retrieval": (
                self.golden_retrieval.to_dict() if self.golden_retrieval is not None else None
            ),
        }


class PreviewPathUnavailable(RuntimeError):
    """The real answer path cannot be located in this process."""


class ReviewPreviewService:
    """Builds the temporary-index preview on top of the real answer path.

    Dependencies are injected so tests run offline; production wiring passes
    the same ``EmbeddingService`` / ``LLMGenerator`` instances the app already
    builds at startup (see ``src/main.py`` lifespan).
    """

    def __init__(
        self,
        store: CanonicalRecordStore,
        *,
        embedder: Any,
        embedding_dim: int,
        generator: Any = None,
        llm: Any = None,
        ner_service: Any = None,
        vectors_path: Optional[Path] = None,
        vectors_dir_name: str = "corpus",
        release_id: str = "preview",
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._embedding_dim = embedding_dim
        self._generator = generator
        self._llm = llm
        self._ner_service = ner_service
        # Serving vectors file, read-only, for content_hash reuse. Tests point
        # it at a real model's vectors file to prove the preview indexes those
        # vectors, not a re-derived copy; None = embed everything.
        self._vectors_path = Path(vectors_path) if vectors_path else None
        self._vectors_dir_name = vectors_dir_name
        self._release_id = release_id

    # -- seams the tests assert on ------------------------------------------

    @staticmethod
    def _synthesize_fn():
        """The real ``POST /v2/query`` synthesis function.

        Resolved from the main module *already loaded in this process* —
        never a fresh import (that would clone the app and compare identity
        against a copy). Under uvicorn the file is ``main``; under pytest it
        is ``src.main``; a script run is ``__main__``.
        """
        for name in ("main", "src.main", "__main__"):
            mod = sys.modules.get(name)
            fn = getattr(mod, "_synthesize_v2_answer", None)
            if callable(fn):
                return fn
        raise PreviewPathUnavailable(
            "no loaded src.main/main module exposes _synthesize_v2_answer - "
            "the preview must run the real answer path, not a private copy"
        )

    def _build_generator(self):
        if self._generator is not None:
            return self._generator
        # The app's generator singleton — the same object main.py assigns to
        # its production global at startup.
        from src.services.llm_generator import get_llm_generator

        return get_llm_generator()

    # -- public API ----------------------------------------------------------

    def diff_for(
        self, record_id: str, version: Optional[int] = None
    ) -> tuple[RecordDiff, CanonicalRecord, tuple[ValidationIssue, ...], str]:
        """(diff, draft, validation issues, status) — no index built.

        Shared by the review-diff endpoint and the full preview so both show
        the reviewer exactly the same diff.
        """
        if version is None:
            draft = self._store.get_record(record_id).latest
        else:
            draft = self._store.get_version(record_id, version)
        baseline = published_version(self._store, record_id)
        diff = diff_records(baseline, draft)
        issues = (
            validate_for_publication(draft)
            if draft.status in PREVIEWABLE_STATUSES
            else ()
        )
        return diff, draft, issues, draft.status.value

    async def preview(
        self,
        record_id: str,
        query: str,
        version: Optional[int] = None,
        top_k: int = 5,
        router_result: Any = None,
    ) -> ReviewPreviewResult:
        """The full review-preview: diff + validation + real-path answer.

        ``router_result`` is a test seam only: passing a pre-computed
        ``RouterResult`` pins the route deterministically (the pipeline's own
        documented parameter). Production leaves it ``None`` = live routing.
        """
        diff, draft, issues, status_note = self.diff_for(record_id, version)
        baseline = published_version(self._store, record_id)
        # T19: anomaly warnings computed from the draft's structured values.
        # peer_values=() keeps the sector-outlier rule silent (no new I/O
        # path); zero-implausible and negative rules still fire.
        values = getattr(draft, "values", None)
        anomaly_warnings: tuple[Anomaly, ...] = (
            find_anomalies(str(draft.record_type), values)
            if isinstance(values, Mapping)
            else ()
        )
        common: dict[str, Any] = dict(
            record_id=record_id,
            draft_version=draft.version,
            baseline_version=baseline.version if baseline else None,
            high_impact_version=HIGH_IMPACT_VERSION,
            diff=diff,
            anomaly_warnings=anomaly_warnings,
        )
        if status_note not in {s.value for s in PREVIEWABLE_STATUSES}:
            # Nothing to preview: a published version IS the live answer, an
            # archived one cannot be submitted. Say so as a field error so the
            # submit button stays blocked on the same screen.
            issues = issues + (
                ValidationIssue(
                    field="status",
                    message=(
                        f"khong the gui duyet: ban {draft.version} dang o "
                        f"trang thai '{status_note}'"
                    ),
                ),
            )
            return ReviewPreviewResult(
                preview_status="blocked_validation",
                validation_errors=issues,
                answer=None,
                answer_status=None,
                model_answer=None,
                chunks=(),
                sources=(),
                **common,
            )
        if issues:
            # Criterion 5: the answer must NOT be produced from data that
            # could not be published. Diff and field errors still return so
            # the reviewer sees what to fix.
            return ReviewPreviewResult(
                preview_status="blocked_validation",
                validation_errors=issues,
                answer=None,
                answer_status=None,
                model_answer=None,
                chunks=(),
                sources=(),
                **common,
            )

        # T05: a draft carrying pattern-derived PII (email / phone /
        # citizen_id) must never be fed to the preview pipeline. The gate
        # scans every unit the chunk builder will index — the prose text
        # AND the structured field units — because a value buried in
        # ``values`` / ``fields`` reaches the external generator the same
        # way body text does. Block before the index is built and before
        # the provider is called; the validation errors name the flag,
        # never the matched value.
        gate_text = "\n".join(
            [draft.text or ""]
            + [unit.value for unit in split_field_units(draft)]
        )
        blocking_flags = blocking_pii_flags(gate_text)
        if blocking_flags:
            return ReviewPreviewResult(
                preview_status="blocked_pii",
                validation_errors=tuple(
                    ValidationIssue(
                        field="draft_text",
                        message=f"van ban co dinh dang PII: {flag}",
                    )
                    for flag in blocking_flags
                ),
                answer=None,
                answer_status=None,
                model_answer=None,
                chunks=(),
                sources=(),
                **common,
            )
        payload, docs, chunks_out, sources_out, golden = await self._run_real_path(
            draft, query, top_k, router_result=router_result
        )
        provider = str(payload.get("provider") or "")
        answer = payload.get("answer")
        model_answer = payload.get("model_answer")
        # Ticket 44: the flag is authoritative, the tag is the safety net.
        blocked = bool(payload.get("generation_failed")) or is_generation_failure(provider)
        if provider == "auto_answer":
            # The router answered without retrieval (CONTACT_BLOCK /
            # HYDE_REJECT_VAGUE). That IS what the student would receive —
            # report it as itself, never as "insufficient evidence".
            preview_status = "auto_answer"
        elif blocked:
            # Provider unavailable (the manifest_303q `generation_blocked`
            # state). The ticket's literal requirement: report the
            # provider-blocked status and STILL return chunks/sources so the
            # reviewer can inspect the grounding — never pretend the answer
            # was verified.
            preview_status = "insufficient_evidence_provider"
            answer = None
        elif not docs:
            preview_status = "insufficient_evidence"
            answer = None
        else:
            preview_status = "ok"
        return ReviewPreviewResult(
            preview_status=preview_status,
            validation_errors=(),
            answer=answer,
            answer_status=provider or None,
            model_answer=bool(model_answer) if model_answer is not None else None,
            chunks=tuple(chunks_out),
            sources=tuple(sources_out),
            golden_retrieval=golden,
            **common,
        )

    # -- temporary index + real answer path ----------------------------------

    async def _run_real_path(
        self,
        draft: CanonicalRecord,
        query: str,
        top_k: int,
        *,
        router_result: Any = None,
    ) -> tuple[dict, list, list, list, Optional[Any]]:
        """Index the draft-as-published in a temp context, run the real path.

        Returns (synthesis payload, docs, chunks, sources, golden report).
        """
        # Windows: an open LanceDB handle can make temp-dir cleanup raise
        # PermissionError; ignore_cleanup_errors keeps the request path clean.
        with tempfile.TemporaryDirectory(
            prefix="husc-preview-", ignore_cleanup_errors=True
        ) as tmp:
            tmp_path = Path(tmp)
            # Everything CPU/disk-bound (materialize, chunk, embed, write the
            # table) runs OFF the event loop — the real EmbeddingService is a
            # sentence-transformers model and would freeze /v2 serving for
            # every in-flight request (main.py wraps every such call).
            built = await asyncio.to_thread(self._build_temp_context, tmp_path, draft)
            retriever, graph = built

            from src.services.graphrag_retriever import (
                GraphRAGRetriever,
                UnifiedRAGPipeline,
            )
            from src.services.ner_service import NERService
            from src.services.query_router import SmartQueryRouter

            # The default NERService writes to a CWD-relative production
            # cache — the preview must never seed draft text into it.
            ner = self._ner_service or NERService(
                llm=self._llm, cache_path=tmp_path / "ner_cache.jsonl"
            )
            pipeline = UnifiedRAGPipeline(
                router=SmartQueryRouter(llm=self._llm),
                graphrag=GraphRAGRetriever(
                    ner_service=ner,
                    graph=graph,
                    chunk_fetcher=retriever.fetch_by_id,
                    chunk_batch_fetcher=retriever.fetch_by_ids,
                ),
                llm=self._llm,
            )
            # 7. Run the answer path exactly as POST /v2/query does:
            # encode -> dense retrieve -> pipeline route/fuse -> synthesize.
            qvec = await asyncio.to_thread(self._embedder.encode_query, query)
            result = await asyncio.to_thread(
                retriever.retrieve,
                query_vector=list(qvec),
                top_k=top_k * 3,
                query=query,
            )
            baseline_docs = result.documents if result.is_success else []
            rag_result = await pipeline.query(
                query, baseline_docs=baseline_docs, top_k=top_k,
                router_result=router_result,
            )
            docs = (
                list(rag_result.documents)
                if rag_result.documents
                else list(baseline_docs)
            )
            # The real generator, per-preview copy: the season/abstention
            # probe (llm_generator._resolve_admission_context) must see the
            # TEMPORARY index, not the production retriever the singleton
            # would otherwise resolve. The singleton itself is never mutated.
            generator = copy.copy(self._build_generator())
            generator.retriever_probe = lambda: retriever
            payload = await self._synthesize_fn()(rag_result, query, generator)
            chunks_out = [
                {
                    "chunk_id": d.chunk_id,
                    "text": d.text,
                    "score": d.score,
                    "source": d.source,
                    "metadata": dict(d.metadata or {}),
                }
                for d in docs
            ]
            sources_out = _dedup_sources(docs)
            golden = await self._golden_check(retriever)
            return payload, docs, chunks_out, sources_out, golden

    async def _golden_check(self, retriever: Any) -> Optional[Any]:
        """T06 layer (d): golden top-5 retrieval over the temp index.

        Pure retrieval, zero LLM calls — one encoded question + one dense
        retrieve per golden question. Advisory only: the report rides along
        in the preview response, it never blocks publication.
        """
        from src.services.golden_retrieval_check import (
            load_golden_set,
            run_golden_retrieval_check,
        )

        golden = load_golden_set()
        if not golden:
            return None
        return await asyncio.to_thread(
            run_golden_retrieval_check, golden, retriever, self._embedder
        )
    def _build_temp_context(self, tmp_path: Path, draft: CanonicalRecord):
        """Steps 1-5 of the temp index (sync; runs inside ``to_thread``)."""
        # 1. Published baseline corpus — the materializer's public API
        #    (ticket 05), into the temp dir.
        materialized = PublishedMaterializer(self._store).materialize(
            self._release_id, tmp_path / "baseline"
        )
        # 2. Substitute the draft, as if published. It passed
        #    validate_for_publication, so the forced-status copy is a record
        #    the model itself accepts.
        corpus = [
            r for r in materialized.admitted if r.record_id != draft.record_id
        ] + [_as_published(draft)]
        # 3. Deterministic chunk identity (ticket 06 public API).
        index = build_chunk_index(
            corpus, tmp_path / self._vectors_dir_name, release_id=self._release_id
        )
        # 4. Vectors: reuse the serving vectors.jsonl by (chunk_id,
        #    content_hash) — the ReindexJob._vectors_for rule — and embed only
        #    what moved (in practice: the draft's chunks).
        texts = [c["text"] for c in index.chunks]
        vectors = self._vectors_for(index.chunks, texts)
        vectors_path = tmp_path / self._vectors_dir_name / "vectors.jsonl"
        vectors_path.parent.mkdir(parents=True, exist_ok=True)
        with vectors_path.open("w", encoding="utf-8") as fh:
            for chunk, vec in zip(index.chunks, vectors):
                fh.write(
                    json.dumps(
                        {
                            "chunk_id": chunk["id"],
                            "content_hash": chunk["content_hash"],
                            "vector": [float(x) for x in vec],
                        }
                    )
                    + "\n"
                )
        # 5. LanceDB rows via the REAL ingestion schema — the same gates
        #    production ingest applies (status stamp, vector/hash cross-check).
        ingest = self._load_ingest()
        rows = ingest.canonical_corpus_rows(
            tmp_path / self._vectors_dir_name,
            embedding_dim=self._embedding_dim,
            stamp=ingest.CorpusStamp.current(),
        )
        lancedb = importlib.import_module("lancedb")
        db = lancedb.connect(str(tmp_path / "lancedb"))
        db.create_table("husc_chunks", rows)
        retriever = LanceDBRetriever(
            LanceDBRetrieverConfig(
                uri=str(tmp_path / "lancedb"),
                table_name="husc_chunks",
                embedding_dim=self._embedding_dim,
            )
        )
        graph = self._build_corpus_graph(index.chunks)
        return retriever, graph

    def _load_ingest(self):
        """``scripts/ingest_lancedb`` under whichever name resolves here.

        Production (uvicorn, cwd=rag2025) has ``rag2025`` on sys.path, so
        ``scripts.ingest_lancedb`` resolves; the test suite additionally has
        the repo root (conftest), where ``rag2025.scripts...`` resolves.
        Neither name may be assumed — and a file-path fallback covers any
        layout where both fail, since re-implementing the row schema locally
        is forbidden (it would skip production's own gates).
        """
        for name in ("scripts.ingest_lancedb", "rag2025.scripts.ingest_lancedb"):
            try:
                return importlib.import_module(name)
            except ModuleNotFoundError:
                continue
        path = _RAG2025_ROOT / "scripts" / "ingest_lancedb.py"
        spec = importlib.util.spec_from_file_location("ingest_lancedb", path)
        if spec is None or spec.loader is None:
            raise PreviewPathUnavailable(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod

    def _vectors_for(self, chunks: list[dict], texts: list[str]) -> list:
        """One vector per chunk: reused from serving where hash matches."""
        reuse: dict[str, tuple[str, Any]] = {}
        if self._vectors_path is not None and self._vectors_path.is_file():
            with self._vectors_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    cid = str(entry.get("chunk_id") or "")
                    if cid:
                        reuse[cid] = (
                            str(entry.get("content_hash") or ""),
                            entry.get("vector"),
                        )
        todo: list[int] = []
        out: list[Any] = [None] * len(chunks)
        for i, (chunk, _text) in enumerate(zip(chunks, texts)):
            hit = reuse.get(str(chunk["id"]))
            if hit is not None and hit[0] == chunk["content_hash"] and hit[1] is not None:
                out[i] = hit[1]
            else:
                todo.append(i)
        if todo:
            fresh = self._embedder.encode_batch([texts[i] for i in todo])
            for i, vec in zip(todo, fresh):
                out[i] = vec
        return out

    def _build_corpus_graph(self, chunks: list[dict]) -> KnowledgeGraph:
        """RECORD->CHUNK graph for the temp corpus (no NER, no production file).

        Node/edge attribute conventions match ``KnowledgeGraphBuilder``'s
        persistence contract: ``chunk_ids`` is a JSON *string* (GraphML has no
        list type — ``KnowledgeGraph.get_chunk_ids_for_entity`` json-decodes
        it), ``schema_version`` is set on the graph, and relation/weight
        mirror the serving edges.
        """
        import networkx as nx

        g = nx.MultiDiGraph()
        g.graph["schema_version"] = "1.0"
        by_record: dict[str, list[dict]] = {}
        for ch in chunks:
            record_id = ch["metadata"]["record_id"]
            by_record.setdefault(record_id, []).append(ch)
        for record_id, chs in sorted(by_record.items()):
            g.add_node(
                f"RECORD:{record_id}",
                text=record_id,
                entity_type="RECORD",
                normalized=record_id.lower(),
                chunk_ids=json.dumps([c["id"] for c in chs]),
            )
            for c in chs:
                g.add_node(
                    c["id"],
                    text=c["text"],
                    entity_type="CHUNK",
                    normalized="",
                    chunk_ids=json.dumps([c["id"]]),
                )
                g.add_edge(
                    f"RECORD:{record_id}",
                    c["id"],
                    key=f"RECORD:{record_id}|{RELATION_HAS_CHUNK}|{c['id']}",
                    relation=RELATION_HAS_CHUNK,
                    weight=1.0,
                    chunk_ids=json.dumps([c["id"]]),
                )
        return KnowledgeGraph(g)


def _dedup_sources(docs) -> list[dict]:
    """Unique source documents behind the returned chunks, in rank order."""
    seen: set[str] = set()
    out: list[dict] = []
    for d in docs:
        meta = d.metadata or {}
        key = str(meta.get("source_document_id") or d.source or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "source_document_id": meta.get("source_document_id"),
                "source": d.source,
                "page": meta.get("source_page"),
                "section": meta.get("section"),
                "title": meta.get("title") or meta.get("section"),
                "record_id": meta.get("record_id"),
                "record_type": meta.get("record_type"),
                "effective_from": meta.get("effective_from"),
                "effective_to": meta.get("effective_to"),
                "authority_level": meta.get("authority_level"),
            }
        )
    return out
