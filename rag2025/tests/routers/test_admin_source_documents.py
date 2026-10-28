"""T05: source-document content byte-serving endpoint.

The service owns the read path; the tests assert the HTTP contract at the seam:

- registered file -> 200, byte-identical body, correct media type.
- path escape (``..``), unregistered id, and a registered file that is gone
  -> 404.
- no credential -> 422 per the admin surface's established convention
  (ticket 13 and 6+ tests in this repo hard-grip 422, not 401).
- a DOCX id serves the DOCX media type and an RFC 5987 Content-Disposition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from rag2025.src.app.lifecycle import (  # noqa: E402
    configure_canonical_record_store,
)
from src.infrastructure.canonical_record_store import CanonicalRecordStore  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402

ACTORS = {"admin": ("cb.quan-tri", "admin")}
TOKENS = {"admin": "tok-admin"}
OPS_TOKEN = "shared-ops-token"


def _headers(role: str) -> dict[str, str]:
    return {"X-Admin-Token": TOKENS[role]}


def _configure(monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_IDENTITIES",
                       '[{"token": "tok-admin", "actor_id": "cb.quan-tri", "role": "admin"}]')
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)


def _app(tmp_path: Path, monkeypatch) -> FastAPI:
    _configure(monkeypatch)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "canonical_store")
    )
    application.state.source_document_repo_root = tmp_path / "repo"
    return application


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    repo = tmp_path / "repo" / "new_data"
    repo.mkdir(parents=True)
    return repo


@pytest.fixture
def client(repo_root: Path, tmp_path: Path, monkeypatch):
    return TestClient(_app(tmp_path, monkeypatch))


def test_content_200_byte_identical(client, repo_root):
    body = b"%PDF-1.4\nminimal pdf content used for byte-echo test\n"
    path = repo_root / "T05_test.pdf"
    path.write_bytes(body)
    reg = client.post("/admin/source-documents/register",
                      json={"document_id": "new_data/T05_test.pdf"}, headers=_headers("admin"))
    assert reg.status_code == 201, reg.text
    resp = client.get("/admin/source-documents/content?document_id=new_data/T05_test.pdf",
                      headers=_headers("admin"))
    assert resp.status_code == 200
    assert resp.content == body
    assert resp.headers["content-type"] == "application/pdf"


def test_content_path_escape_refused(client, repo_root):
    repo_root.joinpath("real.pdf").write_bytes(b"x")
    client.post("/admin/source-documents/register",
                json={"document_id": "new_data/real.pdf"}, headers=_headers("admin"))
    resp = client.get("/admin/source-documents/content?document_id=../real.pdf",
                      headers=_headers("admin"))
    assert resp.status_code == 404
    assert "not in the registry" in resp.json()["detail"]


def test_content_deep_escape_refused(client, repo_root):
    reg = client.post("/admin/source-documents/register",
                      json={"document_id": "new_data/new_data/../../../escape.pdf"},
                      headers=_headers("admin"))
    assert reg.status_code == 422
    resp = client.get("/admin/source-documents/content?document_id=new_data/new_data/../../../escape.pdf",
                      headers=_headers("admin"))
    assert resp.status_code == 404
    assert "not in the registry" in resp.json()["detail"]


def test_content_unregistered_404(client, repo_root):
    resp = client.get("/admin/source-documents/content?document_id=new_data/missing.pdf",
                      headers=_headers("admin"))
    assert resp.status_code == 404
    assert "not in the registry" in resp.json()["detail"]


def test_content_missing_file_404(client, repo_root):
    client.post("/admin/source-documents/register",
                json={"document_id": "new_data/ghost.pdf"}, headers=_headers("admin"))
    resp = client.get("/admin/source-documents/content?document_id=new_data/ghost.pdf",
                      headers=_headers("admin"))
    assert resp.status_code == 404
    assert "is registered but the file is not at" in resp.json()["detail"]


def test_content_no_credential_422(client, repo_root):
    resp = client.get("/admin/source-documents/content?document_id=new_data/T05_test.pdf")
    assert resp.status_code == 422


def test_content_docx_media_type(client, repo_root):
    docx = _make_docx(repo_root / "sample.docx")
    client.post("/admin/source-documents/register",
                json={"document_id": "new_data/sample.docx"}, headers=_headers("admin"))
    resp = client.get("/admin/source-documents/content?document_id=new_data/sample.docx",
                      headers=_headers("admin"))
    assert resp.status_code == 200
    assert resp.headers["content-type"] == \
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert resp.headers["content-disposition"].startswith("inline")
    assert "sample.docx" in resp.headers["content-disposition"]
    assert "filename*=UTF-8" in resp.headers["content-disposition"]


def test_content_txt_media_type(client, repo_root):
    # The registry accepts every supported ingestion extension, so a .txt
    # source must not be served as a Word document — the browser would hand
    # it to the wrong handler and fail to open it.
    body = "Thông báo tuyển sinh 2026\n".encode("utf-8")
    (repo_root / "notice.txt").write_bytes(body)
    client.post("/admin/source-documents/register",
                json={"document_id": "new_data/notice.txt"}, headers=_headers("admin"))
    resp = client.get("/admin/source-documents/content?document_id=new_data/notice.txt",
                      headers=_headers("admin"))
    assert resp.status_code == 200
    assert resp.content == body
    assert resp.headers["content-type"] == "text/plain; charset=utf-8"


def test_content_non_ascii_filename(client, repo_root):
    docx = _make_docx(repo_root / "u1eaf_uaa.docx")
    client.post("/admin/source-documents/register",
                json={"document_id": "new_data/u1eaf_uaa.docx"}, headers=_headers("admin"))
    resp = client.get("/admin/source-documents/content?document_id=new_data/u1eaf_uaa.docx",
                      headers=_headers("admin"))
    assert resp.status_code == 200
    assert resp.headers["content-disposition"].startswith("inline")
    assert "u1eaf" in resp.headers["content-disposition"]
    assert "filename*=UTF-8" in resp.headers["content-disposition"]


def _make_docx(path: Path):
    try:
        import docx
    except ImportError:
        pytest.skip("python-docx not installed")
    d = docx.Document()
    d.add_paragraph("sample docx content")
    d.save(str(path))


def test_content_dotfile_is_404_not_200(client, repo_root):
    """A registered dotfile (``rag2025/.env``) answers 404, not 200 or 500.

    The classifier now refuses dotfile registration outright, so the HTTP
    surface answers 422 on the register call and 404 on the content call —
    neither leaks the file's bytes.
    """
    (repo_root.parent / "rag2025").mkdir(parents=True, exist_ok=True)
    (repo_root.parent / "rag2025" / ".env").write_bytes(b"SECRET=1")
    reg = client.post(
        "/admin/source-documents/register",
        json={"document_id": "rag2025/.env"},
        headers=_headers("admin"),
    )
    # Either the register is refused (classifier, 422) or accepted (pre-change
    # entry, legacy seed); the content call must be 404 in both cases.
    assert reg.status_code in (201, 422), reg.text
    resp = client.get(
        "/admin/source-documents/content?document_id=rag2025/.env",
        headers=_headers("admin"),
    )
    assert resp.status_code == 404


def test_content_gitignore_is_404(client, repo_root):
    """A synthetic ``new_data/.gitignore`` answers 404 on content."""
    (repo_root / ".gitignore").write_bytes(b"*")
    resp = client.get(
        "/admin/source-documents/content?document_id=new_data/.gitignore",
        headers=_headers("admin"),
    )
    assert resp.status_code == 404


def test_content_vietnamese_filename_is_200_with_parseable_header(
    client, repo_root
):
    """A Vietnamese-named file serves 200; the Content-Disposition header
    carries an ASCII-safe fallback plus the true name in ``filename*``."""
    body = "Thông báo tuyển sinh 2026\n".encode("utf-8")
    filename = "báo_cáo_tuyển_sinh_2026.txt"
    (repo_root / filename).write_bytes(body)
    client.post(
        "/admin/source-documents/register",
        json={"document_id": f"new_data/{filename}"},
        headers=_headers("admin"),
    )
    resp = client.get(
        f"/admin/source-documents/content?document_id=new_data/{filename}",
        headers=_headers("admin"),
    )
    assert resp.status_code == 200, resp.text
    assert resp.content == body
    disposition = resp.headers["content-disposition"]
    # The header must be decodable: Starlette encodes headers latin-1, so a
    # non-ASCII `filename="…"` value would have raised UnicodeEncodeError
    # (HTTP 500). Assert both parameters are present and parseable.
    assert "filename=" in disposition
    assert "filename*=UTF-8''" in disposition
    # The ASCII fallback must contain no non-ASCII characters.
    ascii_part = disposition.split(";")[1].strip()
    assert ascii_part.startswith('filename="')
    fallback_value = ascii_part[len('filename="'):-1]
    fallback_value.encode("latin-1")  # must not raise
    # The RFC 5987 parameter carries the true Vietnamese name percent-encoded.
    from urllib.parse import unquote
    star_part = [p for p in disposition.split(";") if p.strip().startswith("filename*=")][0].strip()
    encoded_name = star_part[len("filename*=UTF-8''"):]
    assert unquote(encoded_name) == filename
