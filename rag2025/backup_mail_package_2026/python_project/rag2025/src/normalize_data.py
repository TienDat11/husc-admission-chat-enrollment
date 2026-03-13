"""
Data Normalization Script

Converts mixed raw inputs (.jsonl/.txt/.md/.pdf/.docx) into canonical
record format expected by the chunking pipeline.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterator

from loguru import logger

SUPPORTED_EXTENSIONS = {".jsonl", ".txt", ".md", ".pdf", ".docx"}


def cast_to_int(value):
    """Cast string to int if it represents a valid integer."""
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "doc"


def _stable_doc_id(path: Path, root: Path) -> str:
    rel = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    stem = _slugify(path.stem)
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def _looks_like_legal_header_footer(line: str) -> bool:
    text = line.strip()
    if not text:
        return False

    if re.match(r"^\s*(Trang|Page)\s*\d+(?:\s*/\s*\d+)?\s*$", text, re.IGNORECASE):
        return True

    upper = text.upper()
    if (
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" in upper
        or "Độc lập - Tự do - Hạnh phúc".upper() in upper
        or "ĐỘC LẬP - TỰ DO - HẠNH PHÚC" in upper
    ):
        return True

    if re.match(r"^\s*Số\s*:\s*[^\s]+", text):
        return True

    if re.search(r"(Hà Nội|Huế|TP\.?\s*Hồ Chí Minh|Đà Nẵng).*ngày\s+\d{1,2}\s+tháng\s+\d{1,2}\s+năm\s+\d{4}", text, re.IGNORECASE):
        return True

    return False


def _looks_like_legal_heading(line: str) -> bool:
    text = line.strip()
    return bool(
        re.match(r"^(PHẦN|CHƯƠNG|MỤC)\s+[IVXLC\d]+", text, re.IGNORECASE)
        or re.match(r"^Điều\s+\d+[\.:]", text, re.IGNORECASE)
        or re.match(r"^Khoản\s+\d+[\.:]", text, re.IGNORECASE)
        or re.match(r"^[a-zđ]\)\s+", text, re.IGNORECASE)
    )


def _to_legal_markdown_line(line: str) -> str:
    text = line.strip()
    if re.match(r"^(PHẦN|CHƯƠNG|MỤC)\s+[IVXLC\d]+", text, re.IGNORECASE):
        return f"## {text}"
    if re.match(r"^Điều\s+\d+[\.:]?", text, re.IGNORECASE):
        return f"### {text}"
    if re.match(r"^Khoản\s+\d+[\.:]?", text, re.IGNORECASE):
        return f"- **{text}**"
    if re.match(r"^[a-zđ]\)\s+", text, re.IGNORECASE):
        return f"  - {text}"
    return text


def _clean_legal_text(text: str) -> tuple[str, bool]:
    """Remove repetitive legal header/footer noise while keeping structure markers."""
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return "", False

    counts = Counter(lines)
    max_repeat = max(counts.values()) if counts else 1

    cleaned: list[str] = []
    for line in lines:
        if _looks_like_legal_header_footer(line):
            continue

        # remove repeated boilerplate lines that appear many times
        if counts[line] >= 3 and max_repeat >= 3 and not _looks_like_legal_heading(line):
            continue

        cleaned.append(line)

    if not cleaned:
        return "", False

    out_lines: list[str] = []
    for line in cleaned:
        if _looks_like_legal_heading(line):
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
            out_lines.append(_to_legal_markdown_line(line))
            continue

        out_lines.append(_to_legal_markdown_line(line))

    normalized = "\n".join(out_lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()

    legal_detected = any(_looks_like_legal_heading(ln) for ln in cleaned)
    return normalized, legal_detected


def _normalize_metadata(metadata: dict | None, source: str, source_type: str) -> dict:
    meta = metadata if isinstance(metadata, dict) else {}
    if "source" not in meta:
        meta["source"] = source
    if "source_type" not in meta:
        meta["source_type"] = source_type
    if "year" in meta:
        meta["year"] = cast_to_int(meta["year"])
    return meta


def normalize_record(record: dict, source: str, source_type: str, doc_id: str, line_num: int) -> dict | None:
    """Normalize a single record to canonical pre-chunk format."""
    text = record.get("text") or record.get("text_plain") or record.get("summary")
    if not text or not str(text).strip():
        logger.warning(f"{source}:{line_num} Missing text, skipping")
        return None

    out = dict(record)
    out["doc_id"] = str(out.get("doc_id") or doc_id)

    chunk_id = cast_to_int(out.get("chunk_id", 0))
    if not isinstance(chunk_id, int):
        chunk_id = 0
    out["chunk_id"] = chunk_id

    out["id"] = str(out.get("id") or f"{out['doc_id']}_chunk_{chunk_id}")
    out["text"] = str(text)

    if out.get("text_plain") is not None:
        out["text_plain"] = str(out["text_plain"])
    if out.get("summary") is not None:
        out["summary"] = str(out["summary"])

    out["metadata"] = _normalize_metadata(
        out.get("metadata"),
        source=source,
        source_type=source_type,
    )

    return out


def _extract_text_from_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency pypdf. Please install requirements.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return "\n\n".join(p for p in pages if p)


def _extract_text_from_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Missing dependency python-docx. Please install requirements.") from exc

    doc = Document(str(path))
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n\n".join(parts)


def _iter_input_files(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield input_path
        else:
            logger.warning(f"Unsupported file type: {input_path}")
        return

    if input_path.is_dir():
        for p in sorted(input_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield p
        return

    raise FileNotFoundError(f"Input path not found: {input_path}")


def _records_from_non_jsonl(path: Path, root: Path) -> list[dict]:
    source = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    source_type = path.suffix.lower().lstrip(".")
    doc_id = _stable_doc_id(path, root)

    if path.suffix.lower() in {".txt", ".md"}:
        raw_text = path.read_text(encoding="utf-8", errors="replace")
    elif path.suffix.lower() == ".pdf":
        raw_text = _extract_text_from_pdf(path)
    elif path.suffix.lower() == ".docx":
        raw_text = _extract_text_from_docx(path)
    else:
        return []

    cleaned_text, legal_detected = _clean_legal_text(raw_text)
    text = cleaned_text or raw_text

    metadata = {
        "source": source,
        "source_type": source_type,
    }
    if legal_detected:
        metadata["info_type"] = "van_ban_phap_ly"
        metadata["legal_structure_detected"] = True

    rec = {
        "doc_id": doc_id,
        "id": f"{doc_id}_chunk_0",
        "chunk_id": 0,
        "text": text,
        "text_plain": text,
        "metadata": metadata,
    }
    out = normalize_record(rec, source=source, source_type=source_type, doc_id=doc_id, line_num=1)
    return [out] if out else []


def _records_from_jsonl(path: Path, root: Path) -> list[dict]:
    source = str(path.relative_to(root)) if path.is_relative_to(root) else path.name
    source_type = "jsonl"
    base_doc_id = _stable_doc_id(path, root)
    records: list[dict] = []

    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.error(f"{source}:{line_num} JSON decode error - {e}")
                continue

            if isinstance(obj, str):
                obj = {
                    "doc_id": f"{base_doc_id}_{line_num:04d}",
                    "id": f"{base_doc_id}_{line_num:04d}_chunk_0",
                    "chunk_id": 0,
                    "text": obj,
                    "metadata": {},
                }
            elif not isinstance(obj, dict):
                logger.warning(f"{source}:{line_num} Unsupported record type, skipping")
                continue

            doc_id = str(obj.get("doc_id") or f"{base_doc_id}_{line_num:04d}")
            normalized = normalize_record(
                obj,
                source=source,
                source_type=source_type,
                doc_id=doc_id,
                line_num=line_num,
            )
            if normalized:
                records.append(normalized)

    return records


def normalize_jsonl(input_path: Path, output_path: Path) -> dict:
    """Normalize mixed-format input into canonical JSONL records."""
    stats = {
        "total": 0,
        "normalized": 0,
        "skipped": 0,
        "errors": [],
    }

    root = input_path if input_path.is_dir() else input_path.parent
    logger.info(f"Normalizing {input_path} -> {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for file_path in _iter_input_files(input_path):
            try:
                if file_path.suffix.lower() == ".jsonl":
                    items = _records_from_jsonl(file_path, root)
                else:
                    items = _records_from_non_jsonl(file_path, root)
            except Exception as e:
                logger.error(f"{file_path}: Error - {e}")
                stats["errors"].append({"file": str(file_path), "error": str(e)})
                continue

            stats["total"] += len(items)
            for rec in items:
                if rec is None:
                    stats["skipped"] += 1
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats["normalized"] += 1

    logger.info(f"Normalization complete: {stats['normalized']} records")
    return stats


def main():
    """CLI entry point."""
    if len(sys.argv) < 3:
        print("Usage: python normalize_data.py <input_path(file|dir)> <output.jsonl>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        logger.error(f"Input path not found: {input_path}")
        sys.exit(1)

    stats = normalize_jsonl(input_path, output_path)

    print("\nNormalization Report:")
    print(f"  Total: {stats['total']}")
    print(f"  Normalized: {stats['normalized']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Errors: {len(stats['errors'])}")

    if stats["errors"]:
        print("\nTop 5 Errors:")
        for err in stats["errors"][:5]:
            print(f"  {err}")

    sys.exit(0 if len(stats["errors"]) == 0 else 1)


if __name__ == "__main__":
    main()
