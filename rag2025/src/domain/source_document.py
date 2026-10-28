"""Source-document identity vocabulary — ticket 39.

A published admissions fact cites where it came from with three fields
(``source_document_id`` / ``source_url`` / ``source_page``,
``src/domain/canonical/record.py:124-126``). Until this ticket the citation
only had to be *non-blank*; nothing checked that the string names a document
that exists. This module is the smallest domain layer that makes the citation
a typed thing: what kinds of ids a source document can have, what states a
registry entry moves through, and the one function that decides which kind an
id is.

Decisions recorded here (ticket 39, Bối cảnh + Ghi chú 4):

- ``document_id`` keeps its existing meaning **verbatim**: a repo-relative,
  forward-slash path (``new_data/THONG_TIN_TUYEN_SINH_202611.pdf``), resolved
  by the parser as ``repo_root / document_id``
  (``scripts/seed_canonical_records.py:251``). The registry is a layer *on top*
  of that convention — it does not mint opaque ids and does not migrate
  anything, because ``seed_report.json`` keys its conflict statements by this
  exact string (ticket 38) and 188 stored records already carry it.
- Two id shapes exist in the real store, and they mean different things:
  a bare path to a file (``kind=pdf_document`` — the ticket's mandatory list
  is three PDFs, but the kind names "a real file document", not "PDF" only),
  and a ``*.jsonl#slug`` chunk-corpus pointer (``kind=chunk_corpus``), which
  addresses a slice of a derived corpus, not a document a human uploaded.
  Mixing the two semantics silently is what Ghi chú 4 forbids, so the
  distinction is a function with a name rather than a comment.
- The registry is **registration-by-reference**: adding a source document is
  placing the file in the repo and calling the register endpoint. There is no
  upload route, because ``python-multipart`` is not installed and is not in
  ``requirements.txt`` — see the ticket's Bối cảnh for why that is an
  infrastructure constraint, not a preference.
"""
from __future__ import annotations

from enum import Enum

__all__ = [
    "SOURCE_DOCUMENT_ENTITY",
    "SourceDocumentKind",
    "SourceDocumentState",
    "classify_document_id",
]

# The ``entity`` an audit event carries when the thing acted on is a registry
# entry rather than a canonical record. ``AuditEvent.entity`` is a free string
# (``src/domain/audit.py:157``) whose default is the canonical-record entity
# name; naming our own keeps registry events distinguishable from the
# provenance-move events ticket 12 already emits, without adding an
# ``AuditAction`` (the ticket forbids inventing actions outside
# ``src/domain/audit.py:101-116``).
SOURCE_DOCUMENT_ENTITY = "source_document"


class SourceDocumentKind(str, Enum):
    """What kind of thing a ``document_id`` names.

    ``PDF_DOCUMENT`` is the file-backed kind — the name is kept from the
    ticket's wording ("tài liệu PDF"), and it covers any real file a record
    cites by repo-relative path, not only ``.pdf``. ``CHUNK_CORPUS`` is a
    pointer into a derived ``.jsonl`` corpus (``.../husc_canonical_2026.jsonl
    #husc_nganh_7220104_2026``); the registry deliberately refuses to register
    those (see ``classify_document_id`` and the service docstring) so the
    blind-citation report keeps exactly one meaning: an id the registry does
    not vouch for.
    """

    PDF_DOCUMENT = "pdf_document"
    CHUNK_CORPUS = "chunk_corpus"
    OTHER = "other"


class SourceDocumentState(str, Enum):
    """Lifecycle of a registry entry.

    ``REGISTERED`` with ``file_present=False`` is the state Ghi chú 3 names
    ("registered, file_absent"): the PDFs are gitignored, so a deploy machine
    may legitimately lack the file. That is *observed information*, not an
    error — the entry records what was true at registration time.

    ``RETIRED`` is a soft removal: the entry stays on disk so the audit trail
    can still name what was registered and when. A retired id is not
    "registered" for existence checks; re-registering the same id writes a
    fresh active entry over it.
    """

    REGISTERED = "registered"
    RETIRED = "retired"


def classify_document_id(document_id: str) -> SourceDocumentKind:
    """Which kind of thing ``document_id`` names.

    The rules are the two shapes actually present in the real store (measured
    over ``rag2025/data/canonical_records/records``: 67 distinct ids — 3 file
    paths and 64 ``*.jsonl#slug`` pointers):

    - contains ``#`` and the part before it ends with ``.jsonl``
      -> ``CHUNK_CORPUS`` (a derived-corpus pointer)
    - no ``#``, ends with a file extension, no ``..`` segment, and is not a
      dotfile -> ``PDF_DOCUMENT`` (a file-backed document; hashable when present)
    - anything else -> ``OTHER``

    A *dotfile* is a basename whose only dot is the leading one (``.env``,
    ``.gitignore``): it is not a file document any parser opens, so it is
    classified ``OTHER`` — the registry will never register it, and a blind
    citation of one is reported, not served. Every real document id in the
    store carries a normal extension (``.pdf``, ``.jsonl``), so the three-shape
    contract for real ids is unchanged.

    ``OTHER`` exists so a caller never has to guess whether an odd string was
    meant as a file path; the registry refuses to register it, and the
    blind-citation report still lists records citing it.
    """
    text = (document_id or "").strip()
    if not text:
        return SourceDocumentKind.OTHER

    head, sep, _tail = text.partition("#")
    if sep:
        return (
            SourceDocumentKind.CHUNK_CORPUS
            if head.lower().endswith(".jsonl")
            else SourceDocumentKind.OTHER
        )

    if _is_path_traversal(text):
        return SourceDocumentKind.OTHER
    basename = text.rsplit("/", 1)[-1]
    if "." not in basename:
        # No extension: not a file any parser we ship can open.
        return SourceDocumentKind.OTHER
    # A basename whose only dot is the leading one (``.env``, ``.gitignore``)
    # is a dotfile, not a file document: the registry must not register it.
    if basename.startswith(".") and basename.count(".") == 1:
        return SourceDocumentKind.OTHER
    return SourceDocumentKind.PDF_DOCUMENT


def _is_path_traversal(text: str) -> bool:
    """Whether ``text`` tries to leave the repo root when joined to it.

    ``document_id`` is used as ``repo_root / document_id`` by the parser, so an
    id carrying ``..`` is a path-traversal attempt by construction. Absolute
    paths are rejected by the registry's own containment check; this catches
    the relative escape.
    """
    parts = text.replace("\\", "/").split("/")
    return any(part == ".." for part in parts)
