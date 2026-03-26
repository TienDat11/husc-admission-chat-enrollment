"""
Normalize all chunked_*.jsonl files into a canonical RAGChunk-like shape.

This script rewrites legacy heterogeneous chunk files so every line is an object
with the same attributes used by src/chunker.py output.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger


def _extract_sparse_terms(text: str) -> list[str]:
    stopwords = {
        "và", "các", "của", "có", "được", "cho", "trong", "là", "một", "này",
        "để", "với", "theo", "từ", "đã", "sẽ", "không", "khi", "bằng",
    }
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in stopwords and len(t) > 2][:50]


def _clean_heading(value: str) -> str:
    value = re.sub(r"[_\-]+", " ", value).strip()
    return value[:120] if value else "Chunk"


def _to_markdown_text(text: str, heading: str) -> str:
    body = text.strip()
    if not body:
        return ""
    if body.startswith("#") or "\n## " in body or "\n### " in body:
        return body
    return f"### {_clean_heading(heading)}\n\n{body}"


def _sanitize_doc_id(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^a-zA-Z0-9_]+", "_", value)
    return value.strip("_") or "doc"


def _parse_chunk_id(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.search(r"(\d+)$", value)
        if m:
            return int(m.group(1))
    return default


def _to_canonical(obj: Any, file_name: str, line_num: int) -> dict[str, Any] | None:
    if isinstance(obj, str):
        text = obj.strip()
        if not text:
            return None
        doc_id = _sanitize_doc_id(Path(file_name).stem)
        chunk_id = line_num - 1
        metadata = {"source": file_name, "source_type": "jsonl"}
        text_md = _to_markdown_text(text, f"{doc_id} chunk {chunk_id}")
        return {
            "id": f"{doc_id}_chunk_{chunk_id}",
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": text_md,
            "text_plain": text,
            "summary": None,
            "metadata": metadata,
            "breadcrumbs": [metadata["source"]],
            "prev_chunk_id": "",
            "next_chunk_id": "",
            "sparse_terms": _extract_sparse_terms(text_md),
        }

    if not isinstance(obj, dict):
        return None

    raw_meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
    text = obj.get("text") or obj.get("text_plain") or obj.get("summary") or obj.get("text_raw")
    if not text or not str(text).strip():
        text = json.dumps(obj, ensure_ascii=False)
    if not text or not str(text).strip():
        return None
    text = str(text).strip()

    base_doc = obj.get("doc_id")
    if not base_doc:
        id_hint = str(obj.get("id", ""))
        m = re.match(r"(.+)_chunk_\d+$", id_hint)
        base_doc = m.group(1) if m else Path(file_name).stem
    doc_id = _sanitize_doc_id(str(base_doc))

    chunk_id = _parse_chunk_id(obj.get("chunk_id"), line_num - 1)
    cid = str(obj.get("id") or f"{doc_id}_chunk_{chunk_id}")

    metadata = dict(raw_meta)
    metadata.setdefault("source", file_name)
    metadata.setdefault("source_type", "jsonl")

    faq_type = obj.get("faq_type")
    if faq_type and "faq_type" not in metadata:
        metadata["faq_type"] = faq_type

    breadcrumbs = obj.get("breadcrumbs") if isinstance(obj.get("breadcrumbs"), list) else []
    if not breadcrumbs:
        breadcrumbs = [metadata["source"]]

    sparse_terms = obj.get("sparse_terms") if isinstance(obj.get("sparse_terms"), list) else []

    heading = str(raw_meta.get("name") or obj.get("faq_type") or cid)
    text_md = _to_markdown_text(text, heading)

    if not sparse_terms:
        sparse_terms = _extract_sparse_terms(text_md)

    return {
        "id": cid,
        "doc_id": doc_id,
        "chunk_id": chunk_id,
        "text": text_md,
        "text_plain": str(obj.get("text_plain")) if obj.get("text_plain") is not None else text,
        "summary": str(obj.get("summary")) if obj.get("summary") is not None else None,
        "metadata": metadata,
        "breadcrumbs": breadcrumbs,
        "prev_chunk_id": "",
        "next_chunk_id": "",
        "sparse_terms": [str(t) for t in sparse_terms],
    }


def _link_prev_next(chunks: list[dict[str, Any]]) -> None:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk["doc_id"], []).append(chunk)

    for group in by_doc.values():
        group.sort(key=lambda c: c["chunk_id"])
        for i, chunk in enumerate(group):
            chunk["prev_chunk_id"] = group[i - 1]["id"] if i > 0 else ""
            chunk["next_chunk_id"] = group[i + 1]["id"] if i < len(group) - 1 else ""


def normalize_file(input_path: Path, output_path: Path) -> tuple[int, int]:
    total = 0
    kept = 0
    canonical: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8", errors="replace") as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"{input_path.name}:{line_num} invalid JSON, skipped")
                continue

            out = _to_canonical(obj, input_path.name, line_num)
            if out is not None:
                canonical.append(out)
                kept += 1

    _link_prev_next(canonical)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for row in canonical:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    return total, kept


def normalize_folder(folder: Path, in_place: bool) -> None:
    files = sorted(folder.glob("chunked_*.jsonl"))

    for fp in files:
        out = fp if in_place else folder / "canonical" / fp.name
        total, kept = normalize_file(fp, out)
        logger.info(f"{fp.name}: {kept}/{total} normalized")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize chunked JSONL files to canonical objects")
    parser.add_argument("chunked_dir", type=Path, help="Directory containing chunked_*.jsonl")
    parser.add_argument("--in-place", action="store_true", help="Rewrite original files in place")
    args = parser.parse_args()

    if not args.chunked_dir.exists() or not args.chunked_dir.is_dir():
        raise SystemExit(f"Invalid directory: {args.chunked_dir}")

    normalize_folder(args.chunked_dir, in_place=args.in_place)


if __name__ == "__main__":
    main()
