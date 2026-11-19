"""PDF worker routing and timing (plan §PL-6.8).

The generated-PDF test runs everywhere. The source-PDF test is marked ``slow`` and
``needs_source_pdf`` and reads ``HUSC_SOURCE_PDF_DIR``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.document_parser_worker import _parse_pdf  # noqa: E402


def _build_pdf(pages: list[tuple[list[str], int]]) -> bytes:
    """Minimal PDF: each page has ASCII text lines and ``ruling`` line operators."""
    count = len(pages)
    page_ids = [4 + 2 * i for i in range(count)]
    content_ids = [5 + 2 * i for i in range(count)]
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_id, content_id, (lines, ruling) in zip(page_ids, content_ids, pages):
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        ops = ["BT /F1 9 Tf 20 780 Td 12 TL", *[f"({line}) Tj T*" for line in lines], "ET"]
        ops += [f"20 {700 - 5 * i} m\n500 {700 - 5 * i} l\nS" for i in range(ruling)]
        stream = "\n".join(ops).encode("ascii")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_worker_routes_table_page_and_fails_closed_without_docling(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(
        _build_pdf(
            [
                (
                    ["Chi tieu 7220201 Ky thuat phan mem 120 25.5", "7340101 Quan tri kinh doanh 90 24.75"],
                    12,
                ),
                (["Dieu 1. Pham vi ap dung cua quy che nay cho thi sinh."], 0),
            ]
        )
    )
    result = _parse_pdf(pdf, cap_mb=100)
    assert [page["page"] for page in result["pages"]] == [1, 2]
    assert "7220201" in result["pages"][0]["text"]
    if importlib.util.find_spec("docling") is None:
        assert {b["code"] for b in result["blockers"]} == {"DOCLING_MISSING"}
        assert result["parser"] == "pypdf"


@pytest.mark.slow
@pytest.mark.needs_source_pdf
def test_source_pdf_parses_within_budget() -> None:
    root = os.environ.get("HUSC_SOURCE_PDF_DIR")
    if not root:
        pytest.skip("HUSC_SOURCE_PDF_DIR not set")
    pdf = Path(root) / "Thông tư 06_2026_BGD-ĐT_Quy chế tuyển sinh ĐH năm 2026.pdf"
    if not pdf.exists():
        pytest.skip(f"missing source PDF: {pdf}")
    start = time.perf_counter()
    result = _parse_pdf(pdf, cap_mb=200)
    elapsed = time.perf_counter() - start
    assert elapsed < 180, f"parse took {elapsed:.1f}s"
    assert result["pages"], "no pages parsed"
