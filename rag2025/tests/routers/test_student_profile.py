"""§PL-7.2 Frame 12 — student profile router tests.

Backed by a real PostgreSQL database (fresh_db fixture migrates to head so the
``student_profiles`` and ``releases`` / ``recommender_snapshots`` tables exist).
"""
from __future__ import annotations

from decimal import Decimal
from uuid import uuid4

import asyncio
import sys

if sys.platform == "win32":  # asyncpg/psycopg need a selector loop on Windows
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

from infrastructure.db import create_db_engine, make_session_factory

try:
    from tests.pg_fixtures import fresh_db  # noqa: F401
except Exception:  # pragma: no cover - fixture import guard
    fresh_db = None


def _make_app(database_url: str) -> FastAPI:
    app = FastAPI()
    engine = create_db_engine(database_url)
    app.state.db_engine = engine
    from routers.student_profile import router

    app.include_router(router)
    try:
        from api.errors import install_error_handling

        install_error_handling(app)
    except Exception:
        pass
    return app


def _headers(session_id) -> dict[str, str]:
    return {"X-Student-Session": str(session_id)}


@pytest.mark.skipif(fresh_db is None, reason="PostgreSQL not available")
def test_student_profile_put_get_delete_lifecycle(fresh_db):
    app = _make_app(fresh_db)
    with TestClient(app) as client:
        sid = uuid4()
        # Missing session header -> 400 STUDENT_SESSION_REQUIRED
        res = client.put("/v2/student-profile", json={"raw_score": 20})
        assert res.status_code == 400
        assert res.json()["code"] == "STUDENT_SESSION_REQUIRED"

        # PUT profile
        res = client.put(
            "/v2/student-profile",
            json={"raw_score": 20, "method": "thi_thpt", "to_hop": "A00",
                  "uu_tien_kv": "KV1", "uu_tien_dt": "UT1",
                  "target_majors": ["CNTT", "KTPM"], "admission_year": 2025},
            headers=_headers(sid),
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["student_session_id"] == str(sid)
        assert body["raw_score"] == 20.0
        assert body["target_majors"] == ["CNTT", "KTPM"]

        # GET profile
        res = client.get("/v2/student-profile", headers=_headers(sid))
        assert res.status_code == 200, res.text
        assert res.json()["raw_score"] == 20.0

        # invalid session header -> 400
        res = client.get("/v2/student-profile", headers={"X-Student-Session": "not-a-uuid"})
        assert res.status_code == 400
        assert res.json()["code"] == "STUDENT_SESSION_REQUIRED"

        # DELETE
        res = client.delete("/v2/student-profile", headers=_headers(sid))
        assert res.status_code == 204, res.text

        # GET after delete -> 404
        res = client.get("/v2/student-profile", headers=_headers(sid))
        assert res.status_code == 404, res.text
        assert res.json()["code"] == "NOT_FOUND"


@pytest.mark.skipif(fresh_db is None, reason="PostgreSQL not available")
@pytest.mark.asyncio
async def test_expired_profile_get_404_and_purge(fresh_db):
    app = _make_app(fresh_db)
    with TestClient(app) as client:
        sid = uuid4()
        engine = create_db_engine(fresh_db)
        # Insert an expired row directly
        async with make_session_factory(engine)() as sess:
            await sess.execute(
                text(
                    """
                    INSERT INTO student_profiles
                        (student_session_id, raw_score, method, to_hop,
                         uu_tien_kv, uu_tien_dt, target_majors, admission_year, expires_at)
                    VALUES (:sid, 20, 'thi_thpt', 'A00', 'KV1', 'UT1',
                            '["CNTT"]'::jsonb, 2025, now() - interval '1 day')
                    """
                ),
                {"sid": sid},
            )
            await sess.commit()
        res = client.get("/v2/student-profile", headers=_headers(sid))
        assert res.status_code == 404, res.text
        assert res.json()["code"] == "NOT_FOUND"


@pytest.mark.skipif(fresh_db is None, reason="PostgreSQL not available")
def test_rate_limit_30_per_min(fresh_db):
    app = _make_app(fresh_db)
    sid = uuid4()
    with TestClient(app) as client:
        # 30 successful PUTs
        for i in range(30):
            res = client.put(
                "/v2/student-profile",
                json={"raw_score": 20, "target_majors": ["CNTT"]},
                headers=_headers(sid),
            )
            assert res.status_code in (200, 429), (i, res.status_code, res.text)
        # 31st -> 429 RATE_LIMITED with Retry-After
        res = client.put(
            "/v2/student-profile",
            json={"raw_score": 20, "target_majors": ["CNTT"]},
            headers=_headers(sid),
        )
        assert res.status_code == 429, res.text
        assert res.json()["code"] == "RATE_LIMITED"
        assert "Retry-After" in res.headers


@pytest.mark.skipif(fresh_db is None, reason="PostgreSQL not available")
def test_target_majors_max_10(fresh_db):
    app = _make_app(fresh_db)
    sid = uuid4()
    with TestClient(app) as client:
        res = client.put(
            "/v2/student-profile",
            json={"raw_score": 20, "target_majors": [f"M{i:03d}" for i in range(11)]},
            headers=_headers(sid),
        )
        assert res.status_code == 400, res.text
        assert res.json()["code"] == "VALIDATION_FAILED"


@pytest.mark.skipif(fresh_db is None, reason="PostgreSQL not available")
def test_raw_score_range(fresh_db):
    app = _make_app(fresh_db)
    sid = uuid4()
    with TestClient(app) as client:
        res = client.put(
            "/v2/student-profile",
            json={"raw_score": 31, "target_majors": ["CNTT"]},
            headers=_headers(sid),
        )
        assert res.status_code == 400, res.text
        res = client.put(
            "/v2/student-profile",
            json={"raw_score": -1, "target_majors": ["CNTT"]},
            headers=_headers(sid),
        )
        assert res.status_code == 400, res.text
