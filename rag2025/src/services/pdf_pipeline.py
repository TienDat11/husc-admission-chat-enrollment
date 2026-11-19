"""Three-tier PDF router: pypdf, Docling, then Docling+RapidOCR.

Results cache by SHA-256 file bytes plus ``PARSER_VERSION``. ``pypdf`` is the
text tier and is pinned in requirements.txt. Docling runs only on the pages a
caller selects (``page_range`` / ``docling_pages``), one conversion per page, so
a table page never triggers a whole-file conversion. Docling is imported lazily
and raises RuntimeError when it is missing; callers treat that as a fail-closed
blocker, not a silent fallback.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ParseResult:
    text: str
    tables_md: list
    parser: Literal["pypdf", "docling", "docling-ocr"]
    page_count: int
    needs_review: bool
    page_texts: list = field(default_factory=list)


PARSER_VERSION = "m2-v2"


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


def docling_pages(path: Path, pages: Iterable[int], ocr: bool) -> ParseResult:
    """Docling text for the given 1-based pages, converted one page at a time."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise RuntimeError("Missing dependency docling; table pages cannot be parsed") from exc
    converter = DocumentConverter()
    page_texts: list[tuple[int, str]] = []
    for page in pages:
        document = converter.convert(str(path), page_range=(page, page)).document
        page_texts.append((page, document.export_to_markdown()))
    return ParseResult(
        text="\n\n".join(text for _, text in page_texts if text),
        tables_md=[],
        parser="docling-ocr" if ocr else "docling",
        page_count=len(page_texts),
        needs_review=False,
        page_texts=page_texts,
    )


def _all_pages(path: Path) -> list[int]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency pypdf. Please install requirements.") from exc
    return list(range(1, len(PdfReader(str(path)).pages) + 1))


def _tier1_docling(path: Path, ocr: bool, pages: Sequence[int] | None = None) -> ParseResult:
    return docling_pages(path, pages if pages is not None else _all_pages(path), ocr=False)


def _tier2_docling(path: Path, ocr: bool, pages: Sequence[int] | None = None) -> ParseResult:
    return docling_pages(path, pages if pages is not None else _all_pages(path), ocr=True)


def _needs_escalation(result: ParseResult) -> bool:
    return result.page_count == 0 or len(result.text) / result.page_count < 50


def _needs_ocr(result: ParseResult) -> bool:
    return result.page_count > 0 and len(result.text) / result.page_count < 50


def parse_pdf(
    path: Path,
    cache_dir: Path,
    force: str | None = None,
    page_range: Sequence[int] | None = None,
) -> ParseResult:
    """Parse ``path``. ``page_range`` lists the 1-based pages to read.

    When given, only those pages go to Docling and the pypdf/escalation tiers are
    skipped; ``page_texts`` holds the ``(page, text)`` pairs.
    """
    key = _cache_key(path)
    if page_range is not None:
        pages = list(page_range)
        if not pages or min(pages) < 1:
            raise ValueError(f"invalid page_range {page_range!r}")
        pages_digest = hashlib.sha256(",".join(map(str, pages)).encode("utf-8")).hexdigest()[:16]
        key = f"{key}_pages{pages_digest}"
        cached = _cache_get(cache_dir, key)
        if cached is not None:
            return cached
        if force == "docling-ocr":
            result = _tier2_docling(path, ocr=True, pages=pages)
        else:
            result = _tier1_docling(path, ocr=False, pages=pages)
        _cache_put(cache_dir, key, result)
        return result
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
