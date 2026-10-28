"""Parser worker — runs in its own process (ticket T04).

Invoked by :mod:`src.services.document_parser` as
``python -m src.services.document_parser_worker <path> <kind>
<max_decompressed_mb>`` (``kind`` is ``pdf_document``, ``docx`` or
``xlsx``). Everything that touches the hostile file
happens *here*, never in the chatbot process. The contract with
the parent is one JSON object on the last stdout line; any
internal failure is reported through the same channel (``error``
key), because a crashed worker with no output is indistinguishable
from a parser bug to the parent.
"""
from __future__ import annotations

import struct
import zlib
import io
import json
import sys
import zipfile
from pathlib import Path

from src.services.document_parser import (
    SCAN_PAGE_CHAR_THRESHOLD,
)
from src.services.magic_bytes import DOCX_KIND, PDF_KIND, XLSX_KIND
from src.services.pii_flags import pii_detected


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        _emit(
            error=(
                "usage: document_parser_worker <path> <kind> "
                "<max_decompressed_mb>"
            )
        )
        return 2
    path, kind, cap_argument = argv[1], argv[2], argv[3]
    try:
        cap_mb = int(cap_argument)
    except ValueError:
        _emit(error=f"max_decompressed_mb is not an integer: {cap_argument!r}")
        return 2
    document = Path(path)
    try:
        if kind == PDF_KIND:
            result = _parse_pdf(document, cap_mb)
        elif kind == DOCX_KIND:
            result = _parse_docx(document, cap_mb)
        elif kind == XLSX_KIND:
            result = _parse_xlsx(document, cap_mb)
        else:
            result = {"error": f"unknown kind: {kind!r}"}
    except _Refusal as exc:
        result = {"error": str(exc), "refused": True}
    except Exception as exc:  # hostile file: report, never crash silently
        result = {"error": f"{type(exc).__name__}: {exc}"}
    _emit(**result)
    return 0 if "error" not in result else 1


class _Refusal(Exception):
    """A document the worker refuses to read further (the bomb path)."""


def _parse_pdf(document: Path, cap_mb: int) -> dict:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Missing dependency pypdf. Please install requirements.") from exc

    reader = PdfReader(str(document))
    pages_text: list[str] = []
    scan_pages: list[int] = []
    decompressed = 0
    limit = cap_mb * 1024 * 1024
    for index, page in enumerate(reader.pages):
        decompressed += _page_decompressed_bytes(page)
        if decompressed > limit:
            raise _Refusal(
                f"decompressed content exceeds {cap_mb} MB cap at page "
                f"{index + 1} (decompression bomb refused)"
            )
        text = (page.extract_text() or "").strip()
        pages_text.append(text)
        if len(text) < SCAN_PAGE_CHAR_THRESHOLD:
            scan_pages.append(index + 1)

    text = "\n\n".join(part for part in pages_text if part)
    page_count = len(reader.pages)
    table_count = _pdf_table_count(reader)
    warnings: list[str] = []
    if scan_pages:
        warnings.append(
            f"trang ảnh: {len(scan_pages)}/{page_count} trang "
            f"chỉ có ảnh, không đọc được văn bản "
            f"(trang {', '.join(str(p) for p in scan_pages)})"
        )
    return {
        "text": text,
        "page_count": page_count,
        "chars_per_page": (len(text) / page_count) if page_count else 0.0,
        "parser": "pypdf",
        "scan_pages": scan_pages,
        "pii_detected": pii_detected(text),
        "table_count": table_count,
        "warnings": warnings,
        "pages": [
            {"page": index + 1, "text": pages_text[index]}
            for index in range(page_count)
        ],
        "error": None,
    }


def _page_decompressed_bytes(page) -> int:
    """Inflated size of one page's content stream(s).

    ``get_contents()`` returns a single content stream or an array
    of them; each stream's ``get_data()`` decompresses it. The
    inflated total is what a decompression bomb inflates, so it is
    the quantity the cap is measured in.
    """
    contents = page.get_contents()
    if contents is None:
        return 0
    streams = contents if _is_array(contents) else [contents]
    total = 0
    for stream in streams:
        resolved = stream.get_object() if hasattr(stream, "get_object") else stream
        getter = getattr(resolved, "get_data", None)
        if getter is not None:
            total += len(getter())
    return total


def _is_array(obj) -> bool:
    return hasattr(obj, "__iter__") and not hasattr(obj, "get_data")
#: Chunk size for streaming bounded reads; the cap is checked after
#: each chunk, so worst-case memory overshoot is bounded by this constant.
_READ_CHUNK = 1 << 20

#: Methods the worker can bound the output of.  ``ZIP_STORED`` has no
#: inflation; ``ZIP_DEFLATED`` is bounded by the zlib stream's own EOF.
#: Anything else (bzip2, lzma, ...) is refused up front.
_BOUNDED_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})

#: How much actual output may exceed the declared ``file_size`` before
#: the metadata is flagged as inconsistent (a forged-size bomb signature).
_SIZE_SLACK = 4096


def _local_header(info: zipfile.ZipInfo, handle) -> tuple[int, int, int]:
    """Return ``(name_len, extra_len, method)`` as they appear in the
    member's *local* file header, read from disk.

    CPython's ``ZipInfo`` fields come from the central directory, which a
    hostile writer can desynchronise from the local header on purpose;
    the byte extent of the compressed data must be read from the local
    header, mirroring what ``ZipFile.open`` validates before reading.
    """
    handle.seek(info.header_offset)
    head = handle.read(30)
    if len(head) < 30 or head[:4] != b"PK\x03\x04":
        raise _Refusal(
            f"corrupt local header for {info.filename!r}: "
            "missing or wrong signature"
        )
    # Local header layout (from PK\x03\x04):
    #   0  sig(4)  4  ver(2)  6  flags(2)  8  method(2)  10  time(2)
    #  12  date(2) 14  crc(4) 18  csize(4) 22  usize(4) 26  name_len(2)
    #  28  extra_len(2)
    method = struct.unpack_from("<H", head, 8)[0]
    name_len = struct.unpack_from("<H", head, 26)[0]
    extra_len = struct.unpack_from("<H", head, 28)[0]
    return name_len, extra_len, method


def _inflate_member_bounded(
    handle,
    info: zipfile.ZipInfo,
    cap_mb: int,
    limit: int,
    running_total: int,
) -> tuple[bytes, int]:
    """Stream-decode one member, bounding the *actual* output.

    DEFLATED members are decoded with a raw ``zlib.decompressobj`` fed from
    the local-header data offset, in chunks of ``_READ_CHUNK``.  The
    decompressor's own EOF is the stop signal — the declared ``compress_size``
    is not trusted for read bounds, so a forged tiny ``file_size`` cannot
    truncate the stream (``ZipExtFile.read`` does truncate at ``file_size``).
    The per-member actual output is then checked against the declared size
    and the running total against the cap.

    STORED members are read raw; the declared size *is* the actual byte
    count on disk (no inflation), so reading exactly that many bytes is
    safe and bounded by the file's physical size.

    Returns ``(member_bytes, new_running_total)``.  Raises ``_Refusal``
    when the cap is exceeded or the metadata is inconsistent.
    """
    name_len, extra_len, method = _local_header(info, handle)
    data_start = info.header_offset + 30 + name_len + extra_len

    # The local method field is attacker writable.  It must both be one of
    # the bounded methods *and* agree with the central directory's method —
    # either disagreement means the archive was forged, not merely hostile.
    if method not in _BOUNDED_METHODS or method != info.compress_type:
        raise _Refusal(
            f"unsupported or mismatched compression method "
            f"(local={method}, central={info.compress_type}) on "
            f"{info.filename!r}; only stored/deflated members are bounded"
        )
    if method == zipfile.ZIP_DEFLATED:
        # Raw DEFLATE stream (no zlib wrapper, no gzip header):
        # wbits=-MAX_WBITS decodes the bare stream that ZIP_DEFLATED
        # members store.  One bounded loop regardless of input length:
        # every round feeds at most _READ_CHUNK bytes and checks the cap.
        decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
        actual = bytearray()
        total = running_total
        handle.seek(data_start)
        pending = b""
        while True:
            if pending:
                # Unconsumed input from the previous round: re-feed it
                # with a bounded max_length before reading more from
                # the file, so the decompressor's internal buffer is
                # never re-read past where it stopped consuming.
                out = decompressor.decompress(pending, _READ_CHUNK)
                actual.extend(out)
                total += len(out)
                pending = decompressor.unconsumed_tail
                if total > limit:
                    raise _Refusal(
                        f"decompressed content exceeds {cap_mb} MB cap "
                        f"({total // (1024 * 1024)} MB actual; "
                        "decompression bomb refused)"
                    )
                if decompressor.eof:
                    out = decompressor.decompress(b"", _READ_CHUNK)
                    actual.extend(out)
                    total += len(out)
                    if total > limit:
                        raise _Refusal(
                            f"decompressed content exceeds {cap_mb} MB cap "
                            f"({total // (1024 * 1024)} MB actual; "
                            "decompression bomb refused)"
                        )
                    break
                continue
            chunk = handle.read(_READ_CHUNK)
            if not chunk:
                # File exhausted: feed whatever is left, require a clean
                # stream end.  An unterminated stream is a truncated or
                # hostile archive, not a valid member.
                out = decompressor.decompress(b"")
                actual.extend(out)
                total += len(out)
                if not decompressor.eof:
                    raise _Refusal(
                        f"truncated deflated stream in {info.filename!r}: "
                        "stream end marker not reached "
                        "(corrupt or hostile archive; "
                        "decompression bomb refused)"
                    )
                break
            out = decompressor.decompress(chunk, _READ_CHUNK)
            actual.extend(out)
            total += len(out)
            pending = decompressor.unconsumed_tail
            if total > limit:
                raise _Refusal(
                    f"decompressed content exceeds {cap_mb} MB cap "
                    f"({total // (1024 * 1024)} MB actual; "
                    "decompression bomb refused)"
                )
            if decompressor.eof:
                out = decompressor.decompress(b"", _READ_CHUNK)
                actual.extend(out)
                total += len(out)
                if total > limit:
                    raise _Refusal(
                        f"decompressed content exceeds {cap_mb} MB cap "
                        f"({total // (1024 * 1024)} MB actual; "
                        "decompression bomb refused)"
                    )
                break
    else:
        # ZIP_STORED: no inflation, but a forged small ``file_size`` with a
        # genuine ``compress_size`` is still an inconsistency (the stored
        # raw extent *is* the output extent, so it must agree with the
        # declared size).  Compare both declared extents up front and
        # read the raw bytes in bounded chunks.
        declared_csize = info.compress_size
        declared_usize = info.file_size
        if abs(declared_csize - declared_usize) > _SIZE_SLACK:
            raise _Refusal(
                f"inconsistent zip metadata: {info.filename!r} stores "
                f"{declared_csize} raw bytes but declares "
                f"{declared_usize} uncompressed "
                "(forged size; decompression bomb refused)"
            )
        handle.seek(data_start)
        actual = bytearray()
        total = running_total
        remaining = declared_usize
        while remaining > 0:
            chunk = handle.read(min(_READ_CHUNK, remaining))
            if not chunk:
                break
            actual.extend(chunk)
            total += len(chunk)
            remaining -= len(chunk)
            if total > limit:
                raise _Refusal(
                    f"decompressed content exceeds {cap_mb} MB cap "
                    f"({total // (1024 * 1024)} MB actual; "
                    "decompression bomb refused)"
                )

    member_actual = len(actual)
    if member_actual > info.file_size + _SIZE_SLACK:
        raise _Refusal(
            f"inconsistent zip metadata: {info.filename!r} decompressed to "
            f"{member_actual} bytes but declared {info.file_size} "
            "(forged size; decompression bomb refused)"
        )
    return bytes(actual), total

def _read_members_bounded(
    archive_path: Path,
    cap_mb: int,
) -> dict[str, bytes]:
    """Read every member with bounded inflation, returning the bytes dict.

    Uses raw ``zlib`` for DEFLATED members rather than CPython's
    ``ZipExtFile``: ``ZipExtFile._read1`` truncates the stream to the
    declared ``file_size`` (``self._left``), so a forged tiny declared
    size with a matching prefix CRC would slip past the size check while
    the real stream is hundreds of megabytes.  This helper reads from
    the local-header data offset, trusts only the decompressor's EOF,
    and checks the actual output against the cap and the declared size.
    """
    limit = cap_mb * 1024 * 1024
    members: dict[str, bytes] = {}
    running_total = 0

    with zipfile.ZipFile(str(archive_path)) as archive:
        infos = archive.infolist()
        for info in infos:
            if info.compress_type not in _BOUNDED_METHODS:
                raise _Refusal(
                    f"unsupported compression method {info.compress_type} on "
                    f"{info.filename!r}; only stored/deflated members are bounded"
                )

    with open(archive_path, "rb") as handle:
        for info in infos:
            data, running_total = _inflate_member_bounded(
                handle, info, cap_mb, limit, running_total
            )
            members[info.filename] = data

    return members


def _pdf_table_count(reader) -> int:
    """Tables the PDF exposes, or 0 when the parser cannot tell.

    ``extract_tables`` is best-effort in pypdf; a PDF that errors
    or yields none is reported as zero tables, not as a failure —
    the count is advisory metadata for the draft, not a gate.
    """
    total = 0
    for page in reader.pages:
        extractor = getattr(page, "extract_tables", None)
        if extractor is None:
            return 0
        try:
            total += len(extractor())
        except Exception:
            return 0
    return total


def _parse_docx(document: Path, cap_mb: int) -> dict:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Missing dependency python-docx. Please install requirements.") from exc

    # Bounded read: the cap is decided on *actual* inflated bytes (streamed
    # in chunks, checked after each), not the declared sizes the central
    # directory claims.  The in-memory copy is what the docx parse works
    # from — a cap-passing document is already in RAM (the parent held the
    # whole upload), so the zip is not re-read from disk.
    member_bytes = _read_members_bounded(document, cap_mb)
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in member_bytes.items():
            archive.writestr(name, data)
    archive_buffer.seek(0)
    doc = Document(archive_buffer)

    # Reuse the canonical extractor (paragraphs) so DOCX text reads
    # exactly like the ingestion path's, then append table cell text
    # so a table-only document still yields its content.

    # Collect paragraphs the same way the parser builds text, while
    # recording the offset of any heading inside the final joined text.
    paragraphs: list[str] = []
    headings: list[dict] = []
    running = 0
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        if p.style.name.startswith("Heading ") or p.style.name == "Title":
            headings.append({"text": t, "offset": running})
        paragraphs.append(t)
        running += len(t) + 2
    paragraph_text = "\n\n".join(paragraphs)
    parts: list[str] = []
    if paragraph_text:
        parts.append(paragraph_text)
    table_texts: list[str] = []
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    table_texts.append(cell_text)
    if table_texts:
        parts.append("\n\n".join(table_texts))

    text = "\n\n".join(parts)
    warnings: list[str] = []
    if len(text) < SCAN_PAGE_CHAR_THRESHOLD:
        warnings.append(
            "trang ảnh: tài liệu không chứa văn bản đọc được "
            "(có thể toàn bộ là ảnh)"
        )
    return {
        "text": text,
        "page_count": 1,
        "chars_per_page": float(len(text)),
        "parser": "python-docx",
        "scan_pages": () if len(text) >= SCAN_PAGE_CHAR_THRESHOLD else (1,),
        "pii_detected": pii_detected(text),
        "table_count": len(doc.tables),
        "warnings": warnings,
        "headings": headings,
        "error": None,
    }


def _parse_xlsx(document: Path, cap_mb: int) -> dict:
    """Read the first sheet as a table: row one is the header, the rest are
    data. Each data row comes back as ``{header: value}`` in ``excel_rows``
    plus the same content rendered as readable ``text`` (ticket T17).

    Assumption: the workbook's *first* sheet is the table and its first
    non-empty row is the header (the shape the ticket's "bảng điểm chuẩn /
    chỉ tiêu" files use). Blank cells are dropped from a row's dict.
    """
    try:
        import openpyxl
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency openpyxl. Please install requirements."
        ) from exc

    # Bounded read, same rationale as the docx path above.  The re-zipped
    # stored buffer is what openpyxl parses — the upload probe is written
    # without an extension, so openpyxl sniffs the format from the bytes.
    member_bytes = _read_members_bounded(document, cap_mb)
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in member_bytes.items():
            archive.writestr(name, data)
    archive_buffer.seek(0)
    try:
        workbook = openpyxl.load_workbook(archive_buffer, read_only=True, data_only=True)
        sheet = workbook.worksheets[0]
        iterator = iter(sheet.iter_rows(values_only=True))
        header: tuple[object, ...] = ()
        for header_row in iterator:
            if any(
                cell is not None and str(cell).strip() for cell in header_row
            ):
                header = tuple(_xlsx_scalar(cell) for cell in header_row)
                break
        rows: list[dict] = []
        for row in iterator:
            values = {
                name: _xlsx_scalar(cell)
                for name, cell in zip(header, row)
                if cell is not None and name
            }
            if any(str(value).strip() for value in values.values()):
                rows.append(values)
    finally:
        workbook.close()

    # A readable render for the report/char count; the draft records'
    # ``text`` is built per row by the service from the same dicts.
    text = "\n".join(
        f"{name}: {value}" for values in rows for name, value in values.items()
    )
    warnings: list[str] = []
    if not rows:
        warnings.append("bảng trống: không có dòng dữ liệu; không tạo nháp")
    return {
        "text": text,
        "page_count": 1,
        "chars_per_page": float(len(text)),
        "parser": "openpyxl",
        "scan_pages": (),
        "pii_detected": pii_detected(text),
        "table_count": len(rows),
        "warnings": warnings,
        "headings": (),
        "excel_rows": rows,
        "error": None,
    }


def _xlsx_scalar(cell: object) -> object:
    """A cell value that JSON can carry and text can render.

    JSON-native scalars pass through unchanged; an integer-valued float
    loses its trailing ``.0`` so the rendered text and the payload agree
    (the same rule ``semantic_units._render`` applies); a date or any other
    object becomes its ``str`` so the worker's ``json.dumps`` never fails.
    """
    if cell is None or isinstance(cell, (str, bool, int)):
        return cell
    if isinstance(cell, float):
        return int(cell) if cell.is_integer() else cell
    return str(cell)


def _emit(**result: object) -> None:
    """Print exactly one JSON object — the parent's whole contract."""
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
