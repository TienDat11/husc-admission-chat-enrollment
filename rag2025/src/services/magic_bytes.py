"""Content-type detection by magic bytes — ticket T04.

Uploads are classified by what the bytes *are*, never by the extension: a
``.exe`` renamed ``.pdf`` must not parse as a PDF (acceptance criterion 4).
"""
from __future__ import annotations

#: ``__all__`` and the three kind strings the detector may return, so a caller
#: that compares against the literals sees them all in one place.
__all__ = [
    "PDF_KIND",
    "DOCX_KIND",
    "XLSX_KIND",
    "detect_kind",
]

PDF_KIND = "pdf_document"
DOCX_KIND = "docx"
XLSX_KIND = "xlsx"

_PDF_MAGIC = b"%PDF"
_ZIP_MAGIC = b"PK\x03\x04"
#: DOCX/XLSX are zips that name the OOXML parts in their local file headers,
#: so the member name is findable in the leading bytes of the content.
_DOCX_MEMBER = b"word/document.xml"
_XLSX_MEMBER = b"xl/workbook.xml"


def detect_kind(content: bytes) -> str | None:
    """Which kind of document ``content`` is, from its bytes only.

    - starts with ``%PDF`` -> ``PDF_KIND``
    - starts with the zip local-header magic AND carries the OOXML
      ``word/document.xml`` member name -> ``DOCX_KIND``
    - starts with the zip local-header magic AND carries the OOXML
      ``xl/workbook.xml`` member name -> ``XLSX_KIND``
    - anything else (PE ``MZ``, plain text, other zip formats) -> ``None``
    """
    if content[: len(_PDF_MAGIC)] == _PDF_MAGIC:
        return PDF_KIND
    if content[: len(_ZIP_MAGIC)] == _ZIP_MAGIC:
        if _XLSX_MEMBER in content:
            return XLSX_KIND
        if _DOCX_MEMBER in content:
            return DOCX_KIND
    return None
