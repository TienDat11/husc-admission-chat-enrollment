"""Source-document registry store — ticket 39, criterion 1 + 5.

What this file exists to prove:

- the registry is a first-class object, not a hand-edited JSON file: an id
  registered through the API reads back **byte-identical**, including the
  Vietnamese filename and the exact ``new_data/`` prefix;
- ``document_id`` keeps its meaning: the path the registry resolves is the
  path ``PdfText.__init__`` would open (``repo_root / document_id``), so a
  registry landing cannot silently re-point the parser;
- the state machine (create / update / retire / re-register) keeps
  ``document_id@revision`` naming exactly one entry state;
- an id that can never be a document (a ``*.jsonl#slug`` corpus pointer, a
  ``..`` escape, a blank) is refused on write and answers ``None`` on read —
  it does not raise, because the read path is an existence question asked
  about every citation in the store, and most of them are corpus pointers.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.domain.source_document import (  # noqa: E402
    SourceDocumentKind,
    SourceDocumentState,
    classify_document_id,
)
from src.infrastructure.source_document_registry import (  # noqa: E402
    DOCUMENTS_DIRNAME,
    SOURCE_DOCUMENTS_DIRNAME,
    RegistryPathError,
    RegistryStateError,
    SourceDocumentRegistry,
)

# The constant the ticket names verbatim (seed_canonical_records.py:90).
TTTS_PDF = "new_data/THONG_TIN_TUYEN_SINH_202611.pdf"
VIETNAMESE_PDF = "new_data/Thông tư 06_2026_BGD-ĐT_Quy chế tuyển sinh ĐH năm 2026.pdf"
CORPUS_POINTER = "rag2025/data/chunked/husc_canonical_2026.jsonl#husc_nganh_7220104_2026"

ACTOR = "cb.quan-tri@husc"


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    """A fake checkout with one real file, so hashes mean something."""
    repo = tmp_path / "repo"
    (repo / "new_data").mkdir(parents=True)
    (repo / "new_data" / "THONG_TIN_TUYEN_SINH_202611.pdf").write_bytes(b"%PDF-1.4 ttts")
    (repo / "new_data" / "other.pdf").write_bytes(b"%PDF-1.4 other")
    return repo


@pytest.fixture
def registry(tmp_path: Path, repo_root: Path) -> SourceDocumentRegistry:
    return SourceDocumentRegistry(tmp_path / "store", repo_root)


def _create(reg: SourceDocumentRegistry, document_id: str = TTTS_PDF):
    return reg.create(
        document_id,
        sha256="a" * 64,
        file_present=True,
        actor_id=ACTOR,
        notes="first",
        now=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )


# -- criterion 1: verbatim round trip -----------------------------------------


def test_registered_id_reads_back_byte_identical(registry: SourceDocumentRegistry) -> None:
    """Registering ``TTTS_PDF`` and reading it back returns the exact string.

    The assertion is on equality with the literal the ticket quotes, not with a
    variable, so a normalization (case fold, NFC, slash swap) fails here.
    """
    _create(registry)
    entry = registry.get(TTTS_PDF)
    assert entry is not None
    assert entry.document_id == "new_data/THONG_TIN_TUYEN_SINH_202611.pdf"
    assert entry.kind is SourceDocumentKind.PDF_DOCUMENT
    assert entry.state is SourceDocumentState.REGISTERED
    assert entry.is_registered
    assert entry.revision == 1


def test_vietnamese_id_survives_the_filename_encoding(
    registry: SourceDocumentRegistry,
) -> None:
    """The id with Vietnamese and a comma reads back exactly, not mojibake.

    It also names a file whose *encoded* component cannot collide with the
    un-encoded one: the id is the key, the filename is an accident of storage.
    """
    _create(registry, VIETNAMESE_PDF)
    entry = registry.get(VIETNAMESE_PDF)
    assert entry is not None
    assert entry.document_id == VIETNAMESE_PDF
    files = list((registry.root / DOCUMENTS_DIRNAME).glob("*.json"))
    assert len(files) == 1
    assert files[0].name.startswith("doc-") and files[0].name.endswith("_d.json")


def test_entry_carries_path_and_hash_when_file_exists(
    registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """An entry for a present file records both the path and the hash."""
    _create(registry)
    entry = registry.get(TTTS_PDF)
    assert entry is not None
    assert entry.sha256 == "a" * 64
    assert entry.file_present is True
    assert Path(entry.path) == repo_root / TTTS_PDF

def test_absent_file_is_a_recorded_state_not_an_error(
    registry: SourceDocumentRegistry,
) -> None:
    """``registered, file_absent`` (Ghi chú 3) is observable, not a failure.

    The store persists the observation its caller made — hashing policy lives
    in the service — so an entry with no hash and no file still vouches for
    the id: the registry's job is the id's meaning, not the file's presence.
    """
    registry.create(TTTS_PDF, sha256=None, file_present=False, actor_id=ACTOR)
    entry = registry.get(TTTS_PDF)
    assert entry is not None
    assert entry.file_present is False
    assert entry.sha256 is None
    assert entry.state is SourceDocumentState.REGISTERED
    assert registry.is_registered(TTTS_PDF) is True

# -- criterion 5: the id's meaning is unchanged -------------------------------


def test_resolve_path_is_the_parsers_own_expression(
    registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """``resolve_path`` equals ``repo_root / document_id`` — what PdfText does.

    Compared against the re-derived expression rather than against the stored
    ``path`` field: a test that read the field back could not detect the
    registry starting to mint opaque ids.
    """
    assert registry.resolve_path(TTTS_PDF) == (repo_root / TTTS_PDF).resolve()
    assert registry.resolve_path(TTTS_PDF).is_file()


@pytest.mark.parametrize(
    "document_id",
    [
        "../outside.pdf",
        "new_data/../outside.pdf",
        "/etc/passwd",
        "C:/Windows/win.ini",
        "",
        "   ",
        CORPUS_POINTER,
    ],
)
def test_ids_that_cannot_be_documents_are_refused(
    registry: SourceDocumentRegistry, document_id: str
) -> None:
    """Path traversal, absolute paths, blanks and corpus pointers never land."""
    with pytest.raises(RegistryPathError):
        registry.create(
            document_id, sha256=None, file_present=False, actor_id=ACTOR
        )


def test_unholdable_id_answers_not_registered_instead_of_raising(
    registry: SourceDocumentRegistry,
) -> None:
    """The read path is total: an id the registry cannot hold is simply absent.

    This is the bug the blind-citation report would have hit: 299 of the real
    store's 487 versions cite ``*.jsonl#slug`` pointers, so an existence check
    that raised on them would 500 the report instead of naming them.
    """
    assert registry.is_registered(CORPUS_POINTER) is False
    assert registry.get(CORPUS_POINTER) is None
    assert registry.is_registered("../escape.pdf") is False


# -- state machine -------------------------------------------------------------


def test_update_bumps_revision_and_keeps_registration(
    registry: SourceDocumentRegistry,
) -> None:
    _create(registry)
    entry = registry.update(
        TTTS_PDF,
        sha256="b" * 64,
        file_present=True,
        actor_id="cb.bien-tap@husc",
        now=datetime(2026, 9, 2, tzinfo=timezone.utc),
    )
    assert entry.revision == 2
    assert entry.sha256 == "b" * 64
    assert entry.registered_by == ACTOR  # unchanged by an update
    assert registry.is_registered(TTTS_PDF)


def test_create_on_a_live_entry_is_refused(registry: SourceDocumentRegistry) -> None:
    """Refreshing is ``update``; the trail keeps "started" ≠ "changed"."""
    _create(registry)
    with pytest.raises(FileExistsError):
        registry.create(
            TTTS_PDF, sha256="c" * 64, file_present=True, actor_id=ACTOR
        )


def test_retire_stops_vouching_but_keeps_history(
    registry: SourceDocumentRegistry,
) -> None:
    _create(registry)
    retired = registry.retire(TTTS_PDF, actor_id=ACTOR)
    assert retired.state is SourceDocumentState.RETIRED
    assert retired.revision == 2
    assert registry.is_registered(TTTS_PDF) is False
    assert registry.get(TTTS_PDF) is not None  # still readable
    listed = registry.list_entries()
    assert registry.list_entries(include_retired=True)[0].document_id == TTTS_PDF
    assert listed == []


def test_update_of_a_retired_entry_is_refused(registry: SourceDocumentRegistry) -> None:
    """An edit is not a way to un-retire (revival goes through ``create``)."""
    _create(registry)
    registry.retire(TTTS_PDF, actor_id=ACTOR)
    with pytest.raises(RegistryStateError):
        registry.update(
            TTTS_PDF, sha256="d" * 64, file_present=True, actor_id=ACTOR
        )


def test_revival_continues_the_revision_counter(
    registry: SourceDocumentRegistry,
) -> None:
    """``document_id@revision`` names one state for the life of the id."""
    _create(registry)
    registry.retire(TTTS_PDF, actor_id=ACTOR)
    revived = registry.create(
        TTTS_PDF, sha256="e" * 64, file_present=True, actor_id=ACTOR
    )
    assert revived.revision == 3
    assert revived.state is SourceDocumentState.REGISTERED


def test_update_of_an_absent_id_raises_keyerror(registry: SourceDocumentRegistry) -> None:
    with pytest.raises(KeyError):
        registry.update(
            "new_data/never-registered.pdf",
            sha256="f" * 64,
            file_present=False,
            actor_id=ACTOR,
        )


# -- classification (the one function that decides kind) -----------------------


@pytest.mark.parametrize(
    "document_id,kind",
    [
        (TTTS_PDF, SourceDocumentKind.PDF_DOCUMENT),
        ("a/b/c.docx", SourceDocumentKind.PDF_DOCUMENT),
        (CORPUS_POINTER, SourceDocumentKind.CHUNK_CORPUS),
        ("rag2025/data/chunked/x.jsonl#slug", SourceDocumentKind.CHUNK_CORPUS),
        ("rag2025/data/chunked/x.csv#slug", SourceDocumentKind.OTHER),
        ("de-an-tuyen-sinh-2026", SourceDocumentKind.OTHER),
        ("new_data/../escape.pdf", SourceDocumentKind.OTHER),
        ("", SourceDocumentKind.OTHER),
    ],
)
def test_classify_document_id(document_id: str, kind: SourceDocumentKind) -> None:
    assert classify_document_id(document_id) is kind


def test_list_entries_is_sorted_and_deduplicated(
    registry: SourceDocumentRegistry,
) -> None:
    _create(registry, "new_data/other.pdf")
    _create(registry, TTTS_PDF)
    ids = [entry.document_id for entry in registry.list_entries()]
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 2


# -- upload ids resolve into the store, not the repo ---------------------------


UPLOAD_ID = "uploads/abc123.pdf"


@pytest.fixture
def upload_registry(tmp_path: Path, repo_root: Path) -> SourceDocumentRegistry:
    """A store holding one uploaded blob under ``source_documents/``."""
    store = tmp_path / "store"
    (store / "source_documents").mkdir(parents=True)
    (store / "source_documents" / "abc123.pdf").write_bytes(b"%PDF-1.4 upload")
    return SourceDocumentRegistry(store, repo_root)


def test_upload_id_resolves_under_the_store_not_the_repo(
    upload_registry: SourceDocumentRegistry,
) -> None:
    """``uploads/<name>`` names a blob the uploader wrote under the store root.

    The two id shapes have two different bases; the resolver that used the
    repo base for both made every uploaded citation resolve to a file that
    never exists (the report's "tải lên rồi mà vẫn báo thiếu file").
    """
    path = upload_registry.resolve_path(UPLOAD_ID)
    assert path.is_file()
    assert path.read_bytes() == b"%PDF-1.4 upload"
    assert path.parent.name == SOURCE_DOCUMENTS_DIRNAME
    # And not the repo-relative reading of the same string.
    assert not path.is_relative_to(upload_registry.repo_root)


def test_upload_id_escaping_the_store_is_refused(
    upload_registry: SourceDocumentRegistry,
) -> None:
    """A blob id that climbs out of ``source_documents/`` never resolves."""
    with pytest.raises(RegistryPathError):
        upload_registry.resolve_path("uploads/../../escape.pdf")


def test_repo_relative_ids_still_resolve_against_the_repo(
    upload_registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """The upload fix must not re-point the ordinary ``new_data/`` ids."""
    assert upload_registry.resolve_path(TTTS_PDF) == (repo_root / TTTS_PDF).resolve()
    assert upload_registry.resolve_path(TTTS_PDF).is_file()


# -- is_servable: the byte-serving policy --------------------------------------


def test_is_servable_allows_corpus_root_with_supported_suffix(
    registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """A file under ``new_data/`` with a supported suffix is servable."""
    assert registry.is_servable(TTTS_PDF) is True


def test_is_servable_allows_uploads_ids(registry: SourceDocumentRegistry) -> None:
    """``uploads/…`` is store-relative and always servable."""
    assert registry.is_servable(UPLOAD_ID) is True
def test_is_servable_allows_corpus_pointer_ids(
    registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """A JSONL corpus pointer (``file.jsonl#slug``) is servable: the
    fragment is not part of the file name, and ``.jsonl`` is a supported
    ingestion extension — the real store holds 64 such ids under
    ``rag2025/data/chunked/``.
    """
    assert registry.is_servable(CORPUS_POINTER) is True




def test_is_servable_refuses_repo_root_dotfile(
    registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """``rag2025/.env`` sits outside every allowed corpus root."""
    (repo_root / "rag2025").mkdir(parents=True, exist_ok=True)
    (repo_root / "rag2025" / ".env").write_bytes(b"SECRET=1")
    assert registry.is_servable("rag2025/.env") is False


def test_is_servable_refuses_new_data_dotfile(
    registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """Even under ``new_data/``, a dotfile is refused: it is not a document."""
    (repo_root / "new_data" / ".gitignore").write_bytes(b"*")
    assert registry.is_servable("new_data/.gitignore") is False


def test_is_servable_refuses_unsupported_suffix(
    registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """A file under the corpus root with an unsupported suffix is not servable."""
    (repo_root / "new_data" / "script.py").write_bytes(b"print(1)")
    assert registry.is_servable("new_data/script.py") is False


def test_is_servable_refuses_unknown_repo_path(
    registry: SourceDocumentRegistry, repo_root: Path
) -> None:
    """A repo-relative id that is not under any allowed corpus root."""
    (repo_root / "other").mkdir(parents=True, exist_ok=True)
    (repo_root / "other" / "doc.pdf").write_bytes(b"%PDF")
    assert registry.is_servable("other/doc.pdf") is False
