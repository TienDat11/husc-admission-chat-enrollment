"""Parser worker unit tests: bounded zip reading + PDF page spans.

Covers the P1 ZIP-decompression fix (bounded actual-inflation, forged-size
metadata, unsupported compression methods) and the T05 PDF page-provenance
addition (per-page 1-based page numbers in the parse result).
"""
from __future__ import annotations

import io
import struct
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.document_parser_worker import (  # noqa: E402
    _Refusal,
    _parse_docx,
    _parse_pdf,
    _read_members_bounded,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_pdf(pages: list[str]) -> bytes:
    """Minimal valid PDF: one content stream per page, real xref table.

    Copied from tests/integration/test_admin_document_upload.py so this file
    is self-contained; empty-string pages yield zero extractable characters
    (the "trang ảnh" state pypdf reports).
    """
    n = len(pages)
    page_ids = [4 + 2 * i for i in range(n)]
    content_ids = [4 + 2 * i + 1 for i in range(n)]
    body = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        (
            f"2 0 obj\n<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] "
            f"/Count {n} >>\nendobj\n"
        ).encode(),
        b"3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
    ]
    for page_id, content_id, text in zip(page_ids, content_ids, pages):
        body.append(
            (
                f"{page_id} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>\nendobj\n"
            ).encode()
        )
        stream = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode()
        body.append(
            f"{content_id} 0 obj\n<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream\nendobj\n"
        )
    out = b"%PDF-1.4\n"
    offsets = []
    for obj in body:
        offsets.append(len(out))
        out += obj
    xref_pos = len(out)
    total = len(body)
    xref = f"xref\n0 {total + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode()
    xref += (
        f"trailer\n<< /Size {total + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return out + xref


def _forge_docx_zip(
    payload: bytes,
    declared_file_size: int,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    """Build a one-member DOCX zip and forge the declared ``file_size``
    in both the local and central directory headers.

    The CRC is patched to match the *truncated prefix* so a code path
    that only validates against the declared size still passes; only a
    reader that decodes the full stream catches the inconsistency.
    """
    import zlib

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as zf:
        zf.writestr("word/document.xml", payload)
    raw = bytearray(buf.getvalue())

    # Local file header layout (30 bytes before the name):
    #   0  sig(4)  4  ver(2)  6  flags(2)  8  method(2)  10  time(2)
    #  12  date(2) 14  crc(4) 18  csize(4) 22  usize(4) 26  name_len(2)
    #  28  extra_len(2)
    # Central directory file header layout (46 bytes before the name):
    #   0  sig(4)  4  vmade(2)  6  vneed(2)  8  flags(2)  10  method(2)
    #  12  time(2) 14  date(2) 16  crc(4) 20  csize(4) 24  usize(4)
    #  28  name_len(2) 30  extra_len(2) 32  comment_len(2) 34  disk(2)
    #  36  int_attrs(2) 38  ext_attrs(4) 42  local_offset(4)
    #
    # The CRC is patched to match the *truncated prefix* so a code path
    # that only validates against the declared size still passes; only a
    # reader that decodes the full stream catches the inconsistency.
    prefix = payload[:declared_file_size]
    crc = zlib.crc32(prefix) & 0xFFFFFFFF

    # Patch local header usize and CRC
    lh_idx = raw.find(b"PK\x03\x04")
    if lh_idx != -1:
        struct.pack_into("<I", raw, lh_idx + 22, declared_file_size)
        struct.pack_into("<I", raw, lh_idx + 14, crc)

    # Patch central directory usize and CRC
    cd_idx = raw.find(b"PK\x01\x02")
    if cd_idx != -1:
        struct.pack_into("<I", raw, cd_idx + 24, declared_file_size)
        struct.pack_into("<I", raw, cd_idx + 16, crc)

    result = bytes(raw)
    # Sanity: ZipFile must still open and see the forged declared size
    check_buf = io.BytesIO(result)
    with zipfile.ZipFile(check_buf) as zf:
        assert zf.infolist()[0].file_size == declared_file_size, (
            "fixture did not forge file_size: got "
            f"{zf.infolist()[0].file_size}, want {declared_file_size}"
        )
    return result

# ---------------------------------------------------------------------------
# P1: forged-size DOCX refused
# ---------------------------------------------------------------------------


def test_docx_member_declaring_small_size_inflating_past_cap_is_refused(
    tmp_path: Path,
) -> None:
    """A zip member that declares 1 KB but actually inflates to 3 MB is
    refused by the metadata-consistency check in ``_read_members_bounded``.
    """
    # 3 MiB of highly compressible payload
    payload = b"A" * (3 * 1024 * 1024)
    forged = _forge_docx_zip(payload, declared_file_size=1024)
    path = tmp_path / "forged.docx"
    path.write_bytes(forged)

    # cap of 10 MiB means the cap check itself passes; the metadata check
    # (actual 3 MiB >> declared 1 KiB + slack) must fire
    with pytest.raises(_Refusal, match="inconsistent zip metadata|decompression"):
        _parse_docx(path, cap_mb=10)


def test_docx_cap_exceeded_via_actual_inflation_is_refused(
    tmp_path: Path,
) -> None:
    """A member whose actual inflation exceeds the cap is refused even when
    the declared size is honest (the original bomb path, now checked on
    actual bytes rather than declared bytes).
    """
    # 2 MiB payload, cap of 1 MiB: declared size is honest but actual > cap
    payload = b"B" * (2 * 1024 * 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", payload)
    path = tmp_path / "cap.docx"
    path.write_bytes(buf.getvalue())

    with pytest.raises(_Refusal, match="decompressed content exceeds"):
        _parse_docx(path, cap_mb=1)


def test_docx_unsupported_compression_method_is_refused(
    tmp_path: Path,
) -> None:
    """A member using ZIP_BZIP2 (unbounded) is refused up front."""
    buf = io.BytesIO()
    # zipfile supports ZIP_BZIP2 via the compression parameter (3.11+)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_BZIP2) as zf:
        zf.writestr("word/document.xml", b"<document/>")
    path = tmp_path / "bzip2.docx"
    path.write_bytes(buf.getvalue())

    with pytest.raises(_Refusal, match="unsupported compression method"):
        _parse_docx(path, cap_mb=512)


def test_docx_local_method_forged_is_refused(tmp_path: Path) -> None:
    """A zip whose local header method field is patched to an unbounded
    value (e.g. 14, which is not a real method number) while the central
    directory still says DEFLATED is refused: the local field is attacker
    writable and must agree with the central directory.
    """
    payload = b"D" * 100
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", payload)
    raw = bytearray(buf.getvalue())

    # Patch the local header method field (offset 8) to an invalid value
    lh = raw.find(b"PK\x03\x04")
    if lh != -1:
        struct.pack_into("<H", raw, lh + 8, 14)  # method 14: not stored/deflated

    path = tmp_path / "method_forge.docx"
    path.write_bytes(bytes(raw))

    with pytest.raises(_Refusal, match="unsupported or mismatched compression method"):
        _parse_docx(path, cap_mb=512)


def test_docx_stored_member_with_forged_size_is_refused(tmp_path: Path) -> None:
    """A ZIP_STORED member whose raw (compress_size) extent disagrees with
    its forged ``file_size`` by more than the slack is refused up front:
    stored raw bytes *are* the output bytes, so a mismatch is forged
    metadata, not a legitimate archive.
    """
    payload = b"C" * 1_000_000  # 1 MiB actual raw bytes
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("word/document.xml", payload)
    raw = bytearray(buf.getvalue())

    # Forge both usize fields to 64 KiB (local +22, central +24).
    # compress_size stays honest at 1,000,000, so the stored
    # csize/usize consistency check must fire.
    lh = raw.find(b"PK\x03\x04")
    cd = raw.find(b"PK\x01\x02")
    if lh != -1:
        struct.pack_into("<I", raw, lh + 22, 64 * 1024)
    if cd != -1:
        struct.pack_into("<I", raw, cd + 24, 64 * 1024)

    path = tmp_path / "stored_forge.docx"
    path.write_bytes(bytes(raw))

    with pytest.raises(_Refusal, match="inconsistent zip metadata"):
        _parse_docx(path, cap_mb=512)


def test_legitimate_docx_still_parses(tmp_path: Path) -> None:
    """A cap-passing DOCX generated by python-docx parses successfully."""
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hoc phi nam hoc 2026-2027: Khoa hoc Hue, diem chuan CNTT 25.")
    doc.add_heading("Chi tiet", level=1)
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Nganh"
    table.cell(0, 1).text = "Hoc phi"
    table.cell(1, 0).text = "CNTT"
    table.cell(1, 1).text = "12000000"
    buf = io.BytesIO()
    doc.save(buf)
    path = tmp_path / "ok.docx"
    path.write_bytes(buf.getvalue())

    result = _parse_docx(path, cap_mb=512)
    assert "error" not in result or result.get("error") is None
    assert "Hoc phi" in result["text"]


def test_read_members_bounded_on_normal_zip(tmp_path: Path) -> None:
    """A normal multi-member zip is read correctly by the bounded reader."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.txt", b"hello world " * 100)
        zf.writestr("b.txt", b"x" * 1000)
    path = tmp_path / "multi.zip"
    path.write_bytes(buf.getvalue())

    members = _read_members_bounded(path, cap_mb=512)
    assert len(members) == 2
    assert members["a.txt"] == b"hello world " * 100
    assert members["b.txt"] == b"x" * 1000


# ---------------------------------------------------------------------------
# T05: PDF page provenance
# ---------------------------------------------------------------------------


def test_pdf_pages_carry_correct_1_based_numbers(tmp_path: Path) -> None:
    """A 3-page PDF yields pages=[{page:1,text:..},{page:2,..},{page:3,..}]
    and the joined text is unchanged.
    """
    pdf_bytes = _make_pdf(["Dong thuan A", "Dong thuan B", "Dong thuan C"])
    path = tmp_path / "multi.pdf"
    path.write_bytes(pdf_bytes)

    result = _parse_pdf(path, cap_mb=512)
    pages = result.get("pages")
    assert pages is not None, "pages key missing from parse_pdf result"
    assert len(pages) == 3
    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2
    assert pages[2]["page"] == 3
    # text spans match what the old joined text produced
    joined = "\n\n".join(p["text"] for p in pages if p["text"])
    assert result["text"] == joined
    assert "Dong thuan A" in result["text"]
    assert "Dong thuan B" in result["text"]
    assert "Dong thuan C" in result["text"]


def test_pdf_single_page_span(tmp_path: Path) -> None:
    """A one-page PDF has pages=[{page:1, text:...}] and page_count==1."""
    pdf_bytes = _make_pdf(["Nen dung duy nhat"])
    path = tmp_path / "one.pdf"
    path.write_bytes(pdf_bytes)

    result = _parse_pdf(path, cap_mb=512)
    pages = result.get("pages", [])
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "Nen dung duy nhat" in pages[0]["text"]
    assert result["page_count"] == 1


def test_pdf_empty_page_span_has_empty_text(tmp_path: Path) -> None:
    """An all-scan (empty-text) PDF page still appears in pages with empty
    text so the page number is preserved for citation anchoring.
    """
    pdf_bytes = _make_pdf(["", "", ""])
    path = tmp_path / "scan.pdf"
    path.write_bytes(pdf_bytes)

    result = _parse_pdf(path, cap_mb=512)
    pages = result.get("pages", [])
    assert len(pages) == 3
    assert [p["page"] for p in pages] == [1, 2, 3]
    assert all(p["text"] == "" for p in pages)
    # joined text is empty (no non-empty spans)
    assert result["text"] == ""
