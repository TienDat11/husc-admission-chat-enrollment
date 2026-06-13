"""Tests for the /v2/recommend + /v2/whatif endpoints (PHASE-A4 TDD 1:1).

Uses the SAME importlib pattern as the other router tests, so the heavy
app startup (LanceDB / embeddings / generator) is bypassed: we only need
the endpoint contract to be reachable.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
RAG_ROOT = Path(__file__).resolve().parents[2]
if str(RAG_ROOT) not in sys.path:
    sys.path.insert(0, str(RAG_ROOT))


RECOMMENDER_PATH = SRC / "routers" / "recommender.py"


@pytest.fixture
def rec_module():
    spec = importlib.util.spec_from_file_location("recommender_router_test", RECOMMENDER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["recommender_router_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def client(rec_module):
    app = FastAPI()
    app.include_router(rec_module.router)
    return TestClient(app)


# ---- /v2/recommend contract -----------------------------------------------

def test_recommend_returns_ranked_list(client):
    res = client.post("/v2/recommend", json={"score": 25.0})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["count"] > 0
    assert body["score"] == 25.0
    # Ranked by delta DESC
    deltas = [r["delta"] for r in body["recommendations"]]
    assert deltas == sorted(deltas, reverse=True), f"not sorted: {deltas}"
    # First entry shape
    first = body["recommendations"][0]
    for k in ("major_code", "major_name", "latest_diem_chuan", "latest_year", "delta", "label", "explanation"):
        assert k in first, f"missing {k} in {first}"
    # basis+disclaimer always present
    assert body["basis"]
    assert body["disclaimer"]


def test_recommend_missing_score_returns_clean_error(client):
    res = client.post("/v2/recommend", json={})
    # Pydantic 422 is the contract
    assert res.status_code == 422, res.text
    # Body must be JSON, not a 500
    body = res.json()
    assert "detail" in body


def test_recommend_to_hop_filters_results(client):
    """A00 should keep CNTT; a tổ hợp CNTT doesn't offer must drop it."""
    res = client.post("/v2/recommend", json={"score": 24.0, "to_hop": "A00"})
    assert res.status_code == 200
    codes = {r["major_code"] for r in res.json()["recommendations"]}
    assert "7480201" in codes  # CNTT lists A00
    # Now ask for a tổ hợp CNTT definitely does NOT offer (D07 is Hóa học only)
    res2 = client.post("/v2/recommend", json={"score": 24.0, "to_hop": "D07"})
    assert res2.status_code == 200
    codes2 = {r["major_code"] for r in res2.json()["recommendations"]}
    assert "7480201" not in codes2


# ---- /v2/whatif contract --------------------------------------------------

def test_whatif_returns_p_pass_and_disclaimer(client):
    res = client.post("/v2/whatif", json={"score": 25.0, "major_code": "7480201"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["p_pass"]
    assert body["band"] in ("rất cao", "cao", "trung bình", "thấp")
    assert body["basis"]
    assert body["disclaimer"]
    assert "tham khảo" in body["disclaimer"].lower() or "ước lượng" in body["disclaimer"].lower()


def test_whatif_unknown_major_returns_graceful_blocked(client):
    res = client.post("/v2/whatif", json={"score": 20.0, "major_code": "9999999"})
    assert res.status_code == 200, res.text  # NEVER 500
    body = res.json()
    assert body["band"] == "unknown_major"
    assert body["basis"]
    assert body["disclaimer"]
    assert body["latest_diem_chuan"] is None


def test_whatif_missing_major_code_returns_clean_error(client):
    res = client.post("/v2/whatif", json={"score": 20.0})
    assert res.status_code == 422, res.text


def test_whatif_below_chuan_low_band(client):
    # CNTT 2025 = 17.5; score 15 → delta = -2.5 → "thấp"
    res = client.post("/v2/whatif", json={"score": 15.0, "major_code": "7480201"})
    assert res.status_code == 200
    body = res.json()
    assert body["band"] in ("trung bình", "thấp"), body
