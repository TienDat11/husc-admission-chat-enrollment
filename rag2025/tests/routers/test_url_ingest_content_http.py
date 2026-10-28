"""T21 (spec 02 §1): a URL draft's source document is *servable*.

The ticket's regression: ``url_ingest_service`` wrote the fetched page under
``registry.documents_dir`` with a bare ``url-page-<hash>.html`` id. That id is
neither an ``uploads/`` location nor under an allowed corpus root, so
``SourceDocumentRegistry.is_servable`` refuses it — the draft is registered and
publishable, yet the citation chip on it 404s from ``GET /source-documents/content``.

This is the HTTP seam that catches it: POST ``/source-documents/from-url`` then
GET ``/source-documents/content?document_id=<id>`` and assert the page bytes come
back. The service test asserts ``is_servable`` directly; this one proves the
route the citation chip actually calls answers 200, not 404.

No network: the fetcher is injected on ``app.state.url_ingest_service``.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
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
from src.routers.admin import router as admin_router  # noqa: E402
from src.services.url_ingest_service import (  # noqa: E402
    UrlIngestService,
    document_id_for_url,
)

URL = "https://tuyensinh.husc.edu.vn/thongbao.php?id=74"
PAGE = (
    b"<html><body><main>Thong bao tuyen sinh 2026: "
    b"xet hoc ba tu 23/5/2026 den 14/7/2026</main></body></html>"
)


def _extract(body: bytes) -> str:
    return "Thong bao tuyen sinh 2026: xet hoc ba tu 23/5/2026 den 14/7/2026"


@pytest.fixture
def store_root(tmp_path: Path) -> Path:
    return tmp_path / "canonical_store"


@pytest.fixture
def client(store_root: Path, monkeypatch) -> TestClient:
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps(
            [{"token": "tok-admin", "actor_id": "cb.admin", "role": "admin"}]
        ),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", "shared-ops-token")
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=store_root)
    )
    # A fetch double on the app's seam: no socket, no DNS. The lifecycle
    # service built by ``configure_canonical_record_store`` already carries
    # the registry, so the served page is registered for real.
    application.state.url_ingest_service = UrlIngestService(
        application.state.admin_lifecycle_service,
        fetch=lambda url: PAGE,
        extract=_extract,
        validate=lambda url: url,
        clock=lambda: datetime(2026, 10, 7, tzinfo=timezone.utc),
    )
    with TestClient(application) as test_client:
        yield test_client


def test_url_draft_source_document_is_served(client: TestClient):
    """The fetched page is fetchable through /content — not a 404."""
    headers = {"X-Admin-Token": "tok-admin"}
    response = client.post(
        "/admin/source-documents/from-url",
        json={"url": URL},
        headers=headers,
    )
    assert response.status_code == 201, response.text
    record_id = response.json()["record_id"]

    # the draft carries the store-owned servable id
    stored = client.get(f"/admin/records/{record_id}", headers=headers)
    assert stored.status_code == 200, stored.text
    document_id = stored.json()["source_document_id"]
    assert document_id == document_id_for_url(URL)
    assert document_id.startswith("uploads/")

    # the citation chip's route answers with the page bytes, not 404
    content = client.get(
        "/admin/source-documents/content",
        params={"document_id": document_id},
        headers=headers,
    )
    assert content.status_code == 200, content.text
    assert content.content == PAGE
