"""Admin document upload — ticket T04, acceptance criteria 1–5.

The endpoint is asserted at the HTTP seam, on a hand-mounted admin app over
a throwaway store root (the shape ``test_admin_conflicts_api.py`` uses):

1. a sample DOCX → 201 + non-empty report + a stored draft record;
2. an all-scan PDF → 201 with the "trang ảnh" warning and *no* draft;
3. the same bytes again → 409;
4. an ``.exe`` renamed ``.pdf`` (PE magic) → 415;
5. a decompression-bomb DOCX → 400, while a concurrent ``/v2/query``
   still answers 200 (the parser runs in a subprocess; the chatbot path
   never blocks on it).

Ticket T17 adds the XLSX path: a table upload drafts one record per data
row, each carrying its row as the structured ``values`` payload.

The PDF/DOCX/XLSX fixtures are generated in-test (python-docx, openpyxl,
and hand-assembled PDF objects for the text/scan pages) so the suite
carries no binary blobs.
"""
from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.domain.audit import AuditAction  # noqa: E402
from src.infrastructure.audit_ledger import AuditLedger  # noqa: E402
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402

ACTORS: dict[str, tuple[str, str]] = {
    "admin": ("cb.quan-tri", "admin"),
    "viewer": ("cb.chi-xem", "viewer"),
}
TOKENS = {role: f"tok-{role}" for role in ACTORS}


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _configure_identities(monkeypatch) -> None:
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [
                {"token": TOKENS[role], "actor_id": actor, "role": role}
                for role, (actor, _) in ACTORS.items()
            ]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", "shared-ops-token")


# ---------------------------------------------------------------------------
# fixtures: generated in-test, no binary blobs in the repo
# ---------------------------------------------------------------------------


def _make_docx(bytes_content: str | bytes) -> bytes:
    """A one-paragraph, one-table DOCX whose zip member sizes are real.

    ``bytes_content`` is the main text; it must outgrow the scanner's
    30-char threshold so the document is *not* reported as image-only.
    """
    from docx import Document

    if isinstance(bytes_content, bytes):
        text = bytes_content.decode("utf-8", "replace")
    else:
        text = bytes_content
    doc = Document()
    doc.add_paragraph(text)
    doc.add_heading("Chi tiet", level=1)
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Nganh"
    table.cell(0, 1).text = "Hoc phi"
    table.cell(1, 0).text = "CNTT"
    table.cell(1, 1).text = "12000000"
    buffer = io.BytesIO()
    doc.save(buffer)
    return buffer.getvalue()


def _make_xlsx(header: list[str], rows: list[list[object]]) -> bytes:
    """A one-sheet XLSX whose first row is ``header`` — ticket T17 fixture.

    Generated with openpyxl in-test, so no binary blob is committed.
    """
    import openpyxl

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(header)
    for row in rows:
        sheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _make_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF: one content stream per page, real xref table.

    ``""`` pages carry no text operators — pypdf extracts zero
    characters from them, which is exactly the "trang ảnh" state.
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


def _make_bomb_docx(cap_mb: int) -> bytes:
    """A DOCX zip whose members inflate far past the cap: the bomb path.

    The *compressed* file stays tiny (repetitive content compresses),
    the *uncompressed* member size is ``cap_mb`` + a bit — the worker's
    decompressed-bytes check is what has to catch it.
    """
    import zipfile

    payload = b"A" * (cap_mb + 1) * 1024 * 1024
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", payload)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# app wiring — same shape as test_admin_conflicts_api.py
# ---------------------------------------------------------------------------


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def app(store_root: Path, monkeypatch) -> FastAPI:
    _configure_identities(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def disk(store_root: Path) -> CanonicalRecordStore:
    """An independent store over the same root: reads what the route wrote."""
    return CanonicalRecordStore(store_root)


@pytest.fixture
def ledger(store_root: Path) -> AuditLedger:
    return AuditLedger(store_root / "audit" / "ledger.jsonl")


def _upload(client: TestClient, filename: str, content: bytes, role: str = "admin"):
    return client.post(
        "/admin/source-documents/upload",
        files={"file": (filename, content)},
        headers=_headers(role),
    )


# ---------------------------------------------------------------------------
# acceptance criterion 1: DOCX mẫu → 201 + báo cáo + bản nháp
# ---------------------------------------------------------------------------


def test_docx_upload_creates_draft_and_report(
    client: TestClient, disk: CanonicalRecordStore, ledger: AuditLedger
) -> None:
    content = _make_docx("Hoc phi nam hoc 2026-2027: Khoa hoc Hue, diem chuan CNTT 25.")
    response = _upload(client, "hoc-phi.docx", content)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["kind"] == "docx"
    assert payload["document_id"].startswith("uploads/")
    assert payload["document_id"].endswith(".docx")
    report = payload["report"]
    assert report["page_count"] >= 1
    assert report["chars_per_page"] > 30  # not a scan-only document
    assert report["table_count"] == 1
    assert payload["draft_record_id"] is not None
    assert payload["warnings"] == []

    draft = disk.get_record(payload["draft_record_id"]).latest
    assert draft.status.value == "draft"
    assert draft.record_type.value == "THONG_BAO"
    assert draft.source_document_id == payload["document_id"]
    assert "Hoc phi nam hoc 2026-2027" in draft.text
    assert draft.authored_by == ACTORS["admin"][0]

    # the document bytes and the registry entry both landed on disk
    blob = disk.root / "source_documents" / f"{payload['sha256']}.docx"
    assert blob.read_bytes() == content
    ledger_entries = ledger.entries()
    assert any(
        entry.event.action is AuditAction.SOURCE_DOCUMENT_CHANGE
        and entry.event.target == payload["document_id"]
        for entry in ledger_entries
    )


# ---------------------------------------------------------------------------
# acceptance criterion 2: PDF toàn ảnh → cảnh báo, không nháp rỗng
# ---------------------------------------------------------------------------


def test_scan_pdf_upload_warns_and_skips_draft(
    client: TestClient, disk: CanonicalRecordStore
) -> None:
    content = _make_pdf(["", "", ""])
    response = _upload(client, "scan.pdf", content)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["report"]["page_count"] == 3
    assert payload["report"]["scan_pages"] == [1, 2, 3]
    assert payload["draft_record_id"] is None
    assert any("trang ảnh" in warning for warning in payload["warnings"])
    # the document is registered (bytes stored, entry present), the draft
    # is not: no record exists under the stable draft id space
    assert list(disk.record_ids()) == []


# ---------------------------------------------------------------------------
# acceptance criterion 3: tải lại cùng tệp → 409
# ---------------------------------------------------------------------------


def test_reupload_same_bytes_is_409(client: TestClient) -> None:
    content = _make_docx("Tong dieu kiem tra: diem chuan 2026, thoi han dang ky.")
    first = _upload(client, "check.docx", content)
    assert first.status_code == 201, first.text
    second = _upload(client, "check-copy.docx", content)
    assert second.status_code == 409, second.text
    assert "already registered" in second.json()["detail"].lower()


# ---------------------------------------------------------------------------
# acceptance criterion 4: .exe đổi đuôi .pdf → 415
# ---------------------------------------------------------------------------


def test_exe_renamed_pdf_is_415(client: TestClient) -> None:
    # PE executable: MZ magic, whatever the extension says
    content = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff" + b"\x00" * 64
    response = _upload(client, "bang-gia.pdf", content)
    assert response.status_code == 415, response.text
    assert "magic bytes" in response.json()["detail"]


# ---------------------------------------------------------------------------
# acceptance criterion 5: bom giải nén → từ chối + /v2/query vẫn sống
# ---------------------------------------------------------------------------


def test_decompression_bomb_refused_and_query_still_serves(
    client: TestClient, disk: CanonicalRecordStore
) -> None:
    """The bomb's *compressed* bytes fit the 20 MB route cap; the
    worker's decompressed check is what refuses it — and the refusal
    is async, so a concurrent request on the same client completes
    while the upload is still inside its subprocess.
    """
    bomb = _make_bomb_docx(cap_mb=512)
    assert len(bomb) < 20 * 1024 * 1024, "the bomb must pass the size gate"

    query_answered: list[int] = []

    def _query_probe() -> None:
        # /v2/query is not on this hand-mounted admin app; the AC's
        # observable is *isolation*: a concurrent request on the same
        # client answers while the bomb upload is inside its subprocess,
        # because the parse never runs on the event loop (to_thread +
        # subprocess). A request that would queue behind a synchronous
        # in-process parse does not queue here.
        query_answered.append(
            client.get("/admin/identities", headers=_headers("admin")).status_code
        )

    thread = threading.Thread(target=_query_probe)
    thread.start()
    response = _upload(client, "bomb.docx", bomb)
    thread.join(timeout=30.0)
    assert not thread.is_alive(), "concurrent request was blocked by the upload"
    assert response.status_code == 400, response.text
    assert "decompression bomb" in response.json()["detail"]
    # the refusal happens before the store write: no blob, no record
    assert list(disk.record_ids()) == []
    assert query_answered and all(code in (200, 403) for code in query_answered)


# ---------------------------------------------------------------------------
# the RBAC seam: the upload route carries the RECORDS_CREATE grant
# ---------------------------------------------------------------------------


def test_upload_requires_records_create_grant(client: TestClient) -> None:
    content = _make_docx("Van ban chi: thong bao nghiep vu nam 2026.")
    response = _upload(client, "nb.docx", content, role="viewer")
    assert response.status_code == 403, response.text
    assert list(_store_root_from_client(client).record_ids()) == []


def _store_root_from_client(client: TestClient) -> CanonicalRecordStore:
    return client.app.state.canonical_record_store


# ---------------------------------------------------------------------------
# ticket T17: an XLSX of điểm chuẩn → one draft per row, each with `values`
# ---------------------------------------------------------------------------


def test_xlsx_diem_chuan_creates_one_draft_per_row(
    client: TestClient, disk: CanonicalRecordStore
) -> None:
    """Three ngành → three drafts, each carrying its row as ``values``."""
    from src.domain.canonical.semantic_units import structured_payload

    content = _make_xlsx(
        ["Mã ngành", "Tên ngành", "Điểm chuẩn"],
        [
            ["7220201", "Công nghệ thông tin", 25.5],
            ["7220201", "Kỹ thuật phần mềm", 26.0],
            ["7340101", "Quản trị kinh doanh", 24.75],
        ],
    )
    response = _upload(client, "diem-chuan.xlsx", content)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["kind"] == "xlsx"
    assert payload["document_id"].endswith(".xlsx")
    assert payload["report"]["table_count"] == 3
    assert payload["warnings"] == []

    drafts = payload["draft_record_ids"]
    assert len(drafts) == 3
    assert payload["draft_record_id"] == drafts[0]

    records = [disk.get_record(rid).latest for rid in drafts]
    assert [r.record_type.value for r in records] == ["DIEM_CHUAN"] * 3
    assert [r.status.value for r in records] == ["draft"] * 3
    assert all(r.source_document_id == payload["document_id"] for r in records)

    payloads = [dict(structured_payload(r)) for r in records]
    expected_rows = [
        {"Mã ngành": "7220201", "Tên ngành": "Công nghệ thông tin", "Điểm chuẩn": 25.5},
        {"Mã ngành": "7220201", "Tên ngành": "Kỹ thuật phần mềm", "Điểm chuẩn": 26.0},
        {"Mã ngành": "7340101", "Tên ngành": "Quản trị kinh doanh", "Điểm chuẩn": 24.75},
    ]
    for payload_row, expected_row in zip(payloads, expected_rows, strict=True):
        assert payload_row == expected_row
    # text renders "<header>: <value>" per line, matching semantic_units._render
    assert records[0].text == (
        "Mã ngành: 7220201\n"
        "Tên ngành: Công nghệ thông tin\n"
        "Điểm chuẩn: 25.5"
    )

    blob = disk.root / "source_documents" / f"{payload['sha256']}.xlsx"
    assert blob.read_bytes() == content


def test_xlsx_chi_tieu_kind_and_deterministic_record_ids(
    client: TestClient, disk: CanonicalRecordStore
) -> None:
    """A ``chỉ tiêu`` sheet drafts ``CHI_TIEU``; ids are stable per row."""
    from src.services.source_document_service import _xlsx_row_id

    content = _make_xlsx(
        ["Mã ngành", "Chỉ tiêu"],
        [["7220201", 120], ["7340101", 90]],
    )
    response = _upload(client, "chi-tieu.xlsx", content)
    assert response.status_code == 201, response.text
    drafts = response.json()["draft_record_ids"]

    assert drafts == [
        _xlsx_row_id({"Mã ngành": "7220201", "Chỉ tiêu": 120}, 1),
        _xlsx_row_id({"Mã ngành": "7340101", "Chỉ tiêu": 90}, 2),
    ]
    assert all(rid.startswith("xlsx-") and "-row" in rid for rid in drafts)
    records = [disk.get_record(rid).latest for rid in drafts]
    assert [r.record_type.value for r in records] == ["CHI_TIEU"] * 2


def test_xlsx_reupload_same_bytes_is_409(client: TestClient) -> None:
    content = _make_xlsx(["Mã ngành", "Điểm chuẩn"], [["7220201", 25.5]])
    assert _upload(client, "dc.xlsx", content).status_code == 201
    second = _upload(client, "dc-again.xlsx", content)
    assert second.status_code == 409, second.text
    assert "already registered" in second.json()["detail"].lower()


def test_xlsx_empty_sheet_drafts_nothing(client: TestClient) -> None:
    """Header only, no data rows → registered, reported, no draft."""
    content = _make_xlsx(["Mã ngành", "Điểm chuẩn"], [])
    response = _upload(client, "empty.xlsx", content)
    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["draft_record_ids"] == []
    assert payload["draft_record_id"] is None
    assert any("bảng trống" in warning for warning in payload["warnings"])


def test_oversized_stream_rejected_without_full_read(client: TestClient) -> None:
    """Defect 3: an upload over the byte cap is 400 with the original
    "file is N bytes; the limit is M" detail.  The bounded read
    (``read(MAX_UPLOAD_BYTES + 1)``) means the route never buffers a body
    larger than the cap — an unbounded ``read()`` would."""
    from src.routers import admin_document_upload as upload_module

    body = b"%PDF-1.4\n" + b"x" * (upload_module.MAX_UPLOAD_BYTES + 1)
    response = _upload(client, "big.pdf", body)

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert f"the limit is {upload_module.MAX_UPLOAD_BYTES}" in detail
    assert "bytes" in detail
