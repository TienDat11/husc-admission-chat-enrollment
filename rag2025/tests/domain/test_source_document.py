"""Source-document domain vocabulary — ticket 39.

One function decides what a ``source_document_id`` *means*: whether the string
names a file in the repo, a pointer into a chunk corpus, or neither. Everything
downstream (registry entry kind, blind-citation counts, what the router refuses
to register) is a consequence of this classification, so the table is pinned
here rather than re-derived in each caller's test.

The corpus-pointer shape is not decoration: 299 of the real store's 487
versions cite ``<file>.jsonl#<slug>`` (``data/chunked/husc_canonical_2026.jsonl``
plus a semantic-unit slug). Those are citations of a *derived* corpus, not of a
document a human uploaded, and conflating the two silently is what makes a
provenance report unreadable.

This file owns the *shape rule* only. Whether the registry accepts a shape it
can see is a second, separate check — see
``tests/infrastructure/test_source_document_registry.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.source_document import (  # noqa: E402
    SOURCE_DOCUMENT_ENTITY,
    SourceDocumentKind,
    classify_document_id,
)


@pytest.mark.parametrize(
    "document_id,kind",
    [
        # File-backed documents: the repo-relative paths the seed writes.
        (
            "new_data/THONG_TIN_TUYEN_SINH_202611.pdf",
            SourceDocumentKind.PDF_DOCUMENT,
        ),
        (
            "new_data/2304_BGDDT-GDDH_HD tuyển sinh ĐH, CĐ 2026.pdf",
            SourceDocumentKind.PDF_DOCUMENT,
        ),
        (
            "new_data/Thông tư 06_2026_BGD-ĐT_Quy chế tuyển sinh ĐH năm 2026.pdf",
            SourceDocumentKind.PDF_DOCUMENT,
        ),
        # The kind names "a real file document", not "PDF" only: the parser's
        # own extension list (``src/chunker.py`` PdfText) covers these too.
        ("docs/handbook.docx", SourceDocumentKind.PDF_DOCUMENT),
        ("a/b/c/x.xlsx", SourceDocumentKind.PDF_DOCUMENT),
        # A bare filename is a repo-relative path like any other — it resolves
        # to ``repo_root/<name>``, where the file may or may not sit. Ghi chú 3
        # makes "no file here" a recorded state rather than a rejected id, so
        # the shape rule must not pre-judge it.
        ("THONG_TIN_TUYEN_SINH_202611.pdf", SourceDocumentKind.PDF_DOCUMENT),
        # Chunk-corpus pointers.
        (
            "rag2025/data/chunked/husc_canonical_2026.jsonl#husc_nganh_7220104_2026",
            SourceDocumentKind.CHUNK_CORPUS,
        ),
        ("data/chunked/x.jsonl#slug", SourceDocumentKind.CHUNK_CORPUS),
        # Everything else: a bare slug from the pre-registry era, an id that
        # escapes the repo root when joined to it, a non-file pointer, a
        # dotfile (only dot is the leading one), or nothing at all.
        ("de-an-tuyen-sinh-2026", SourceDocumentKind.OTHER),
        ("rag2025/data/chunked/x.csv#slug", SourceDocumentKind.OTHER),
        ("new_data/../outside.pdf", SourceDocumentKind.OTHER),
        ("../outside.pdf", SourceDocumentKind.OTHER),
        ("new_data/../../outside.pdf", SourceDocumentKind.OTHER),
        ("new_data\\..\\..\\outside.pdf", SourceDocumentKind.OTHER),
        # Dotfiles: a basename whose only dot is the leading one is not a
        # file document any parser opens.
        (".env", SourceDocumentKind.OTHER),
        (".gitignore", SourceDocumentKind.OTHER),
        ("rag2025/.env", SourceDocumentKind.OTHER),
        ("new_data/.env", SourceDocumentKind.OTHER),
        # A second dot is a real extension, not a dotfile.
        (".env.local", SourceDocumentKind.PDF_DOCUMENT),
        ("", SourceDocumentKind.OTHER),
        ("   ", SourceDocumentKind.OTHER),
    ],
)
def test_classify_document_id(document_id: str, kind: SourceDocumentKind) -> None:
    assert classify_document_id(document_id) is kind


def test_pointer_wins_over_extension_when_both_are_present() -> None:
    """A ``.jsonl#slug`` is a pointer even though it also carries an extension.

    The two shapes are mutually exclusive by construction, and the order of the
    checks is what makes it so: reading the extension first would classify a
    corpus pointer as a document and register a derived corpus as if a human
    had uploaded it.
    """
    assert classify_document_id("x.jsonl#slug") is SourceDocumentKind.CHUNK_CORPUS


def test_classification_is_case_insensitive_on_the_extension() -> None:
    assert classify_document_id("new_data/A.PDF") is SourceDocumentKind.PDF_DOCUMENT
    assert classify_document_id("new_data/a.JsonL#s") is (
        SourceDocumentKind.CHUNK_CORPUS
    )


def test_traversal_is_the_only_shape_the_classifier_refuses_as_a_path() -> None:
    """Absolute paths are a *file shape*; the registry is what refuses them.

    Splitting the checks this way is deliberate: ``..`` is meaningless once a
    caller joins the id to a root, so the shape rule rejects it, while
    ``/etc/passwd.pdf`` names a real file and only the registry's containment
    check can say whether *this* checkout may read it. A caller that trusted
    the classifier alone would still be refused at registration; a caller that
    trusted only the registry would never have classified an escape as a
    document. See the infrastructure test that registers neither.
    """
    assert classify_document_id("/etc/passwd.pdf") is SourceDocumentKind.PDF_DOCUMENT
    assert classify_document_id("C:/Windows/win.ini") is SourceDocumentKind.PDF_DOCUMENT
    assert classify_document_id("../outside.pdf") is SourceDocumentKind.OTHER


def test_registry_events_are_told_apart_by_entity_not_by_action() -> None:
    """The audit ``entity`` marker for a registry event.

    Ticket 39 adds no ``AuditAction`` (the enum is closed to what tickets 12-38
    need), so ``entity`` is the only thing that separates "this document was
    registered" from "a record's citation moved".
    """
    assert SOURCE_DOCUMENT_ENTITY == "source_document"
