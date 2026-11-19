"""A1 baseline measurement (plan U1, Verification). Read-only.

Writes eval/baseline_A1.json. Nothing is written to the LanceDB directory.

Retrieval: LanceDBRetriever (src/services/lancedb_retrieval.py) over table
``husc_chunks`` in --lancedb-dir. Questions are embedded in ONE batched call that
uses the same inputs as EmbeddingService.encode_query (Qwen prompt "query",
normalized), because per-query encoding costs ~11 s each on CPU.

Ground truth (tests/eval/fixtures/golden_seed.jsonl):
  in_scope rows: expected_coordinate_keys. A retrieved chunk matches key
  RT:scope:subject:year:round:methods when its program code (metadata program_code
  or major_code, or the 7-digit code inside its doc_id) equals subject[:7] AND its
  data_year equals year. The chunk table has no record-type or page column, so this
  is a coarse proxy; the rule is recorded in the JSON.
  citation_precision uses document_id only: the chunk table has no page column.
  out_of_scope rows: expect_abstain=true.

Run (main checkout data; Qwen embedder/tokenizer from the offline HF cache):
  HF_HUB_OFFLINE=1 python eval/run_baseline.py --lancedb-dir <main>/rag2025/data/lancedb \
      --source-pdf-dir <main>/new_data --out eval/baseline_A1.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

GOLDEN_SEED = ROOT / "tests" / "eval" / "fixtures" / "golden_seed.jsonl"
CHUNK_PROFILES = ROOT / "config" / "chunk_profiles.yaml"
QWEN_TOKENIZER = "Qwen/Qwen3-Embedding-0.6B"
TABLE = "husc_chunks"
TOP_K = 10
CITATION_K = 5
RECALL_KS = (5, 10)
SAMPLE_SIZE = 200
TT06_PDF = "Thông tư 06_2026_BGD-ĐT_Quy chế tuyển sinh ĐH năm 2026.pdf"
TTS_PDF = "THONG_TIN_TUYEN_SINH_202611.pdf"
_CODE_PREFIX = re.compile(r"^\d{7}")
_CODE_IN_ID = re.compile(r"(?<!\d)(\d{7})(?!\d)")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ratio(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def _load_seed(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _embed_questions(embedder, questions: list[str], expected_dim: int) -> dict:
    """One batched encode with the same inputs as EmbeddingService.encode_query."""
    embedder._ensure_loaded()
    unique = list(dict.fromkeys(questions))
    vectors = embedder._model.encode(
        unique,
        prompt_name=embedder._query_prompt_name(),
        normalize_embeddings=embedder.normalize,
        convert_to_numpy=True,
    ).astype("float32")
    if vectors.shape[1] != expected_dim:
        raise ValueError(f"query embedding dim {vectors.shape[1]} != {expected_dim}")
    return {q: vectors[i].tolist() for i, q in enumerate(unique)}


@dataclass(frozen=True)
class _Facts:
    codes: frozenset[str]
    year: int | None


def _facts(metadata: dict, doc_id: str) -> _Facts:
    codes = set(_CODE_IN_ID.findall(doc_id))
    for field in ("program_code", "major_code"):
        value = metadata.get(field)
        if isinstance(value, str) and _CODE_PREFIX.match(value):
            codes.add(value[:7])
    raw_year = metadata.get("data_year", metadata.get("year"))
    try:
        year: int | None = int(raw_year)
    except (TypeError, ValueError):
        year = None
    return _Facts(frozenset(codes), year)


def _matches(facts: _Facts, key: str) -> bool:
    parts = key.split(":")
    subject, year = parts[2], int(parts[3])
    return facts.year == year and subject[:7] in facts.codes


def _doc_ids(lancedb_dir: Path) -> dict[str, str]:
    import lancedb

    table = lancedb.connect(str(lancedb_dir)).open_table(TABLE).to_arrow()
    rows = table.select(["id", "doc_id"]).to_pylist()
    return {row["id"]: row["doc_id"] for row in rows}


def _retrieve(retriever, vector: list[float], question: str, doc_ids: dict[str, str]) -> list[tuple[_Facts, str, float]]:
    result = retriever.retrieve(query_vector=vector, top_k=TOP_K, query=question)
    if result.error_type is not None:
        raise RuntimeError(f"retrieval failed for {question!r}: {result.error_type} {result.error_message}")
    hits: list[tuple[_Facts, str, float]] = []
    for doc in result.documents[:TOP_K]:
        doc_id = doc_ids.get(doc.chunk_id, "")
        hits.append((_facts(doc.metadata or {}, doc_id), doc_id, float(doc.score)))
    return hits


def _measure(seed: list[dict], retriever, vectors: dict, doc_ids: dict[str, str], threshold: float) -> dict:
    expected_total = 0
    hits_at = {k: 0 for k in RECALL_KS}
    citation_returned = 0
    citation_relevant = 0
    in_scope_top1: list[float] = []
    in_scope_rows = [row for row in seed if row["group"] == "in_scope"]

    for row in in_scope_rows:
        keys = row["expected_coordinate_keys"]
        hits = _retrieve(retriever, vectors[row["question"]], row["question"], doc_ids)
        expected_total += len(keys)
        for k in RECALL_KS:
            top = hits[:k]
            hits_at[k] += sum(1 for key in keys if any(_matches(f, key) for f, _, _ in top))
        top_cited = hits[:CITATION_K]
        citation_returned += len(top_cited)
        citation_relevant += sum(
            1 for facts, _, _ in top_cited if any(_matches(facts, key) for key in keys)
        )
        if hits:
            in_scope_top1.append(hits[0][2])

    out_rows = [row for row in seed if row["group"] == "out_of_scope"]
    abstained = 0
    out_top1: list[float] = []
    for row in out_rows:
        hits = _retrieve(retriever, vectors[row["question"]], row["question"], doc_ids)
        top1 = hits[0][2] if hits else 0.0
        out_top1.append(top1)
        if top1 < threshold:
            abstained += 1

    return {
        "recall_at_5": _ratio(hits_at[5], expected_total),
        "recall_at_10": _ratio(hits_at[10], expected_total),
        "citation_precision": _ratio(citation_relevant, citation_returned),
        "abstention_rate": _ratio(abstained, len(out_rows)),
        "detail": {
            "in_scope_rows": len(in_scope_rows),
            "expected_coordinate_keys": expected_total,
            "hits_at_5": hits_at[5],
            "hits_at_10": hits_at[10],
            "citation_returned_at_5": citation_returned,
            "citation_relevant_at_5": citation_relevant,
            "out_of_scope_rows": len(out_rows),
            "abstained": abstained,
            "abstention_threshold": threshold,
            "top1_score_p50": {
                "in_scope": round(statistics.median(in_scope_top1), 4) if in_scope_top1 else None,
                "out_of_scope": round(statistics.median(out_top1), 4) if out_top1 else None,
            },
        },
    }


def _sample_chunks(lancedb_dir: Path) -> list[str]:
    import lancedb

    table = lancedb.connect(str(lancedb_dir)).open_table(TABLE).to_arrow()
    rows = table.select(["id", "text"]).to_pylist()
    rows.sort(key=lambda r: r["id"])
    texts = [r["text"] for r in rows if r.get("text")]
    return texts[:SAMPLE_SIZE]


class _TokenDriftUnavailable(Exception):
    """The tokenizers cannot be loaded here; the run records null plus the reason."""


def _token_drift(texts: list[str]) -> dict:
    try:
        import tiktoken
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise _TokenDriftUnavailable(f"not_computed: missing dependency: {exc}") from exc
    if not texts:
        raise _TokenDriftUnavailable("not_computed: no chunk texts sampled")
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        qwen = AutoTokenizer.from_pretrained(QWEN_TOKENIZER)
    except Exception as exc:  # e.g. OSError on an offline HF cache miss
        raise _TokenDriftUnavailable(f"not_computed: {exc}") from exc
    cl_counts = [len(enc.encode(t)) for t in texts]
    qwen_counts = [len(qwen.encode(t, add_special_tokens=False)) for t in texts]
    drifts = [abs(q - c) / c for q, c in zip(qwen_counts, cl_counts) if c]
    return {
        "method": "cl100k_base vs Qwen3-Embedding tokenizer, first 200 non-empty husc_chunks texts by id",
        "sample_size": len(texts),
        "cl100k_mean_tokens": round(statistics.mean(cl_counts), 2),
        "qwen_mean_tokens": round(statistics.mean(qwen_counts), 2),
        "mean_abs_relative_drift": round(statistics.mean(drifts), 4) if drifts else None,
        "max_abs_relative_drift": round(max(drifts), 4) if drifts else None,
    }


def _chunk_stats(pdf_dir: Path) -> dict:
    from pypdf import PdfReader
    import tiktoken

    from src.chunker import ChunkConfig, Chunker

    chunker = Chunker(ChunkConfig(CHUNK_PROFILES))
    enc = tiktoken.get_encoding("cl100k_base")
    stats: dict[str, object] = {}
    for label, name in (("tt06", TT06_PDF), ("tts", TTS_PDF)):
        path = pdf_dir / name
        if not path.exists():
            stats[label] = f"not_computed: source PDF missing: {path}"
            continue
        text = "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
        doc = {"id": label, "text": text, "metadata": {"source_file": name, "data_year": 2026}}
        chunks = chunker.chunk_document(doc, profile_name="auto")
        tokens = sorted(len(enc.encode(c.text)) for c in chunks)
        n = len(tokens)
        stats[label] = {
            "source": name,
            "extractor": "pypdf",
            "chunker": "src/chunker.py Chunker.chunk_document(profile=auto)",
            "n_chunks": n,
            "tokens_p50": statistics.median(tokens) if tokens else None,
            "tokens_max": tokens[-1] if tokens else None,
            "share_ge_30_tokens": _ratio(sum(1 for t in tokens if t >= 30), n),
            "chunks_over_512_tokens": sum(1 for t in tokens if t > 512),
        }
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A1 baseline (plan U1).")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "baseline_A1.json")
    parser.add_argument("--seed", type=Path, default=GOLDEN_SEED)
    parser.add_argument("--lancedb-dir", type=Path, default=ROOT / "data" / "lancedb")
    parser.add_argument(
        "--source-pdf-dir",
        type=Path,
        default=Path(os.environ.get("HUSC_SOURCE_PDF_DIR") or ROOT / "new_data"),
    )
    args = parser.parse_args(argv)

    from config.settings import settings
    from services.embedding import EmbeddingService
    from services.lancedb_retrieval import LanceDBRetriever, LanceDBRetrieverConfig

    seed = _load_seed(args.seed)
    retriever = LanceDBRetriever(
        LanceDBRetrieverConfig(
            uri=str(args.lancedb_dir),
            table_name=TABLE,
            embedding_dim=settings.EMBEDDING_DIM,
            default_top_k=TOP_K,
        )
    )
    embedder = EmbeddingService(settings)
    questions = [row["question"] for row in seed if row["group"] in ("in_scope", "out_of_scope")]
    vectors = _embed_questions(embedder, questions, settings.EMBEDDING_DIM)
    retrieval = _measure(
        seed, retriever, vectors, _doc_ids(args.lancedb_dir), settings.RAG_CONF_THRESHOLD
    )
    token_drift, token_drift_reason = None, None
    try:
        token_drift = _token_drift(_sample_chunks(args.lancedb_dir))
    except _TokenDriftUnavailable as exc:
        token_drift_reason = str(exc)
    chunk_stats = _chunk_stats(args.source_pdf_dir)

    result = {
        "recall_at_5": retrieval["recall_at_5"],
        "recall_at_10": retrieval["recall_at_10"],
        "citation_precision": retrieval["citation_precision"],
        "abstention_rate": retrieval["abstention_rate"],
        "chunk_stats": chunk_stats,
        "token_drift": token_drift,
        "token_drift_reason": token_drift_reason,
        "detail": retrieval["detail"],
        "definitions": {
            "recall_at_k": "matched expected coordinate keys / expected keys, micro over in_scope rows, top-k chunks",
            "key_match": "chunk program code (program_code/major_code or 7-digit code in doc_id) == subject[:7] AND data_year == key year",
            "citation_precision": "top-5 chunks of in_scope rows that match >=1 expected key / top-5 chunks returned; document_id level, no page column",
            "abstention_rate": "out_of_scope rows whose top-1 retrieval score < settings.RAG_CONF_THRESHOLD / out_of_scope rows",
            "chunk_stats": "current src/chunker.py run on pypdf text of the TT06 and TTS source PDFs",
        },
        "seed": {"path": str(args.seed.relative_to(ROOT)).replace("\\", "/"), "rows": len(seed)},
        "embedder": {
            "model": settings.EMBEDDING_MODEL,
            "dim": settings.EMBEDDING_DIM,
            "query_prompt": embedder._query_prompt_name(),
            "batched": True,
            "tokenizer": QWEN_TOKENIZER,
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        },
        "lancedb": {"uri": str(args.lancedb_dir), "table": TABLE},
        "git_sha": _git_sha(),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({k: result[k] for k in ("recall_at_5", "recall_at_10", "citation_precision", "abstention_rate")}, ensure_ascii=False))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
