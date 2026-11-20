"""Record ids that contain "/" resolve on every admin record route.

The regulation-derived ids (``DIEU_KIEN:06/2026/TT-BGDĐT:dieu_1:2026:none:all``)
contain a slash. The frontend percent-encodes it (``%2F``), and Starlette decodes
that before routing, so each record-id route must accept slashes. Ordering is part
of the contract: the bare ``GET /records/{record_id:path}`` must come after the
suffixed GET routes, and the preview router must be mounted before the admin
router, or the bare route swallows ``review-diff``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Match

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import RAGSettings  # noqa: E402
from src.app.lifecycle import configure_canonical_record_store  # noqa: E402
from src.routers.admin import router as admin_router  # noqa: E402

MAIN_PATH = ROOT / "src" / "main.py"
SLASH_ID = "DIEU_KIEN:06/2026/TT-BGDĐT:dieu_1:2026:none:all"
PLAIN_ID = "CHI_TIEU:nganh:7220104:2026"
EDITOR_TOKEN = "tok-editor"
EDITOR_ACTOR = "cb.bien-tap"
OPS_TOKEN = "shared-ops-token"

# (method, path template, handler name) for every route that takes a record id.
RECORD_ROUTES = (
    ("GET", "/admin/records/{rid}", "get_record"),
    ("GET", "/admin/records/{rid}/versions", "get_record_versions"),
    ("POST", "/admin/records/{rid}/edits", "edit_record"),
    ("POST", "/admin/records/{rid}/claim", "claim_record"),
    ("POST", "/admin/records/{rid}/transitions", "transition_record"),
    ("GET", "/admin/records/{rid}/review-diff", "review_diff"),
    ("POST", "/admin/records/{rid}/review-preview", "review_preview"),
)


def _resolved_handler(app, method: str, path: str) -> str | None:
    """Name of the first route that fully matches, in the order Starlette tries them."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": "",
        "headers": [],
        "query_string": b"",
    }
    for route in app.routes:
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return route.endpoint.__name__
    return None


@pytest.fixture(scope="module")
def real_app():
    """The app exactly as ``src/main.py`` mounts it. Routing only: no lifespan, no warmup."""
    spec = importlib.util.spec_from_file_location("husc_real_main_slashes", MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.app


@pytest.mark.parametrize("record_id", [SLASH_ID, PLAIN_ID], ids=["slash", "plain"])
@pytest.mark.parametrize(("method", "template", "handler"), RECORD_ROUTES)
def test_record_route_resolves_to_its_handler(real_app, record_id, method, template, handler):
    path = template.format(rid=record_id)
    assert _resolved_handler(real_app, method, path) == handler


@pytest.fixture
def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv(
        "ADMIN_IDENTITIES",
        json.dumps([{"token": EDITOR_TOKEN, "actor_id": EDITOR_ACTOR, "role": "editor"}]),
    )
    monkeypatch.setenv("ADMIN_API_TOKEN", OPS_TOKEN)
    application = FastAPI()
    application.include_router(admin_router)
    configure_canonical_record_store(
        application, settings=RAGSettings(CANONICAL_STORE_ROOT=tmp_path / "store")
    )
    with TestClient(application) as test_client:
        yield test_client


def _record(record_id: str) -> dict:
    return {
        "record_id": record_id,
        "record_type": "DIEU_KIEN",
        "version": 1,
        "admission_year": 2026,
        "effective_from": "2026-01-01",
        "effective_to": "2026-12-31",
        "status": "draft",
        "authority_level": "husc_official",
        "source_document_id": "de-an-tuyen-sinh-2026",
        "source_url": "https://tuyensinh.husc.edu.vn/de-an-2026.pdf",
        "source_page": 24,
        "text": "Điều kiện dự tuyển.",
    }


def test_slash_id_round_trips_through_the_real_http_routes(client):
    headers = {"X-Admin-Token": EDITOR_TOKEN}
    created = client.post("/admin/records", json={"record": _record(SLASH_ID)}, headers=headers)
    assert created.status_code == 200, created.text

    # The frontend sends %2F; a raw slash must also work.
    for rid in (quote(SLASH_ID, safe=""), SLASH_ID):
        got = client.get(f"/admin/records/{rid}", headers=headers)
        assert got.status_code == 200, got.text
        assert got.json()["record_id"] == SLASH_ID

        versions = client.get(f"/admin/records/{rid}/versions", headers=headers)
        assert versions.status_code == 200, versions.text

    submitted = client.post(
        f"/admin/records/{quote(SLASH_ID, safe='')}/transitions",
        json={"action": "submit", "expected_version": 1},
        headers=headers,
    )
    assert submitted.status_code == 200, submitted.text
    assert submitted.json()["record"]["status"] != "draft"
