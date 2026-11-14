"""Error envelope (plan PL-7) and X-Request-ID propagation, on a minimal app."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.errors import install_error_handling


def _app() -> FastAPI:
    app = FastAPI()
    install_error_handling(app)

    @app.get("/missing")
    def missing():
        raise HTTPException(status_code=404, detail="Not Found")

    @app.get("/frozen")
    def frozen():
        raise HTTPException(
            status_code=503,
            detail={"code": "WRITES_FROZEN", "message": "Ghi tạm dừng", "reason": "schema_revision"},
            headers={"Retry-After": "5"},
        )

    @app.get("/needs-int")
    def needs_int(q: int):
        return {"q": q}

    return app


def test_http_error_uses_plan_envelope_and_echoes_request_id() -> None:
    client = TestClient(_app())
    res = client.get("/missing", headers={"X-Request-ID": "req-abc-123"})
    assert res.status_code == 404
    body = res.json()
    assert body["code"] == "NOT_FOUND"
    assert body["field_errors"] == []
    assert body["request_id"] == "req-abc-123"
    assert res.headers["X-Request-ID"] == "req-abc-123"
    assert body["detail"] == "Not Found"  # legacy key kept for existing clients


def test_generated_request_id_when_header_missing_or_invalid() -> None:
    client = TestClient(_app())
    res = client.get("/missing")
    assert res.headers["X-Request-ID"] and res.json()["request_id"] == res.headers["X-Request-ID"]

    res = client.get("/missing", headers={"X-Request-ID": "bad id with spaces"})
    assert res.json()["request_id"] == res.headers["X-Request-ID"] != "bad id with spaces"


def test_validation_error_uses_plan_shape_with_field_errors() -> None:
    client = TestClient(_app())
    res = client.get("/needs-int", params={"q": "not-a-number"})
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "VALIDATION_FAILED"
    assert body["request_id"] == res.headers["X-Request-ID"]
    fields = [e["field"] for e in body["field_errors"]]
    assert "q" in fields, body


def test_503_keeps_detail_reason_and_retry_after_ms() -> None:
    client = TestClient(_app())
    res = client.get("/frozen")
    assert res.status_code == 503
    body = res.json()
    assert body["code"] == "WRITES_FROZEN"
    assert body["detail"]["reason"] == "schema_revision"
    assert body["retry_after_ms"] == 5000
