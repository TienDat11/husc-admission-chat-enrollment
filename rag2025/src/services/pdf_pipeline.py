"""Three-tier PDF router: pypdf, Docling, then Docling+RapidOCR.

Results cache by SHA-256 file bytes plus ``PARSER_VERSION``. ``pypdf`` is not
listed in requirements.txt (only ``neo4j>=5.14.0``) and is unavailable in the
active environment; lazy import preserves module import and raises the same
RuntimeError as ``rag2025/src/normalize_data.py:174`` when invoked. Docling
CPU baseline is about 2.1 seconds per page, suitable for offline ingest.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ParseResult:
    text: str
    tables_md: list
    parser: Literal["pypdf", "docling", "docling-ocr"]
    page_count: int
    needs_review: bool


PARSER_VERSION = "m2-v1"


def _cache_key(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"{digest}_{PARSER_VERSION}"


def _cache_get(cache_dir: Path, key: str) -> ParseResult | None:
    cache_path = cache_dir / f"{key}.json"
    if not cache_path.exists():
        return None
    return ParseResult(**json.loads(cache_path.read_text(encoding="utf-8")))


def _cache_put(cache_dir: Path, key: str, result: ParseResult) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8"
    )


def _tier0_pypdf(path: Path) -> ParseResult:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency pypdf. Please install requirements.") from exc
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return ParseResult(
        text="\n\n".join(page for page in pages if page),
        tables_md=[],
        parser="pypdf",
        page_count=len(reader.pages),
        needs_review=False,
    )


def _tier1_docling(path: Path, ocr: bool) -> ParseResult:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(path))
    document = result.document
    page_count = len(document.pages) if hasattr(document, "pages") else 0
    return ParseResult(
        text=document.export_to_markdown(),
        tables_md=[],
        parser="docling",
        page_count=page_count,
        needs_review=False,
    )


def _tier2_docling(path: Path, ocr: bool) -> ParseResult:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(path))
    document = result.document
    page_count = len(document.pages) if hasattr(document, "pages") else 0
    return ParseResult(
        text=document.export_to_markdown(),
        tables_md=[],
        parser="docling-ocr",
        page_count=page_count,
        needs_review=False,
    )


def _needs_escalation(result: ParseResult) -> bool:
    return result.page_count == 0 or len(result.text) / result.page_count < 50


def _needs_ocr(result: ParseResult) -> bool:
    return result.page_count > 0 and len(result.text) / result.page_count < 50


def parse_pdf(path: Path, cache_dir: Path, force: str | None = None) -> ParseResult:
    key = _cache_key(path)
    if force is None:
        cached = _cache_get(cache_dir, key)
        if cached is not None:
            return cached
    if force == "pypdf":
        result = _tier0_pypdf(path)
    elif force == "docling":
        result = _tier1_docling(path, ocr=False)
    elif force == "docling-ocr":
        result = _tier2_docling(path, ocr="rapidocr")
    else:
        result = _tier0_pypdf(path)
        if _needs_escalation(result):
            result = _tier1_docling(path, ocr=False)
            if _needs_ocr(result):
                result = _tier2_docling(path, ocr="rapidocr")
    _cache_put(cache_dir, key, result)
    return result
