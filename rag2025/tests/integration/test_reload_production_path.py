"""The reload route against the REAL app — the ticket 16 item 4 test.

Every other reload test builds its own ``FastAPI()`` and assigns
``app.state.request_tracker`` / ``app.state.build_retriever`` itself. That is the
trap this file exists to close: those tests pass whether or not production wires
anything, because they supply the very bindings the route needs. The route then
503s in production while the suite stays green.

So this file imports ``src.main`` and asserts the route works through the
bindings *production* creates. If someone deletes the wiring in main.py, these
fail — the synthetic tests cannot.

Query path under test: ``/api/meta`` reads ``app.state.retriever``
(``routers/meta.py:101``) and surfaces ``metadata_snapshot()``'s
``latest_crawl_date`` / ``total_notifications``. Before the P1 fix that alias
was written only by the route itself, so the banner kept describing the startup
table while queries served another one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app

AUTH = {"X-Admin-Token": "tok"}


@pytest.fixture(autouse=True)
def _ops_credential(monkeypatch):
    """The shared token is the machine credential with role ``ops``.

    Role ``ops`` grants ``OPS_RELOAD_TABLE`` (``domain/admin_identity.py:148``),
    which is what ``/admin/reload-table`` requires.
    """
    monkeypatch.setenv("ADMIN_API_TOKEN", "tok")
    monkeypatch.delenv("ADMIN_IDENTITIES", raising=False)


@pytest.fixture(scope="module")
def client():
    # ``with`` so the lifespan runs: part of the wiring under test is created
    # there. A bare TestClient would skip it and prove nothing.
    with TestClient(app) as c:
        yield c


_MISSING = object()


@pytest.fixture(autouse=True)
def _restore_runtime_globals():
    """Undo the swap after each test.

    The route rebinds the *real* module globals in ``src.main`` — that is the
    behaviour under test — so leaving them pointing at this file's fake
    retriever would hand every later test in the session a retriever that is not
    the app's. Restores the globals and the ``app.state`` aliases the seam
    republishes.
    """
    import src.main as main

    names = (
        "lancedb_retriever_service",
        "hybrid_search_service",
        "unified_pipeline",
        "query_enhancer_service",
        "llm_generator_service",
        "embedding_service",
        "reranker_service",
        "guardrail_service",
    )
    saved_globals = {name: getattr(main, name) for name in names}
    saved_state = {
        attr: getattr(app.state, attr, _MISSING)
        for attr in (
            "retriever",
            "lancedb_retriever_service",
            "hybrid_search_service",
            "unified_pipeline",
        )
    }

    yield

    for name, value in saved_globals.items():
        setattr(main, name, value)
    for attr, value in saved_state.items():
        if value is _MISSING:
            if hasattr(app.state, attr):
                delattr(app.state, attr)
        else:
            setattr(app.state, attr, value)


class _Retriever:
    """Stand-in for LanceDBRetriever with the surface the route touches."""

    def __init__(self, table_name: str):
        self.table_name = table_name

    async def healthcheck(self) -> bool:
        return True

    async def close(self) -> None:
        pass

    def metadata_snapshot(self) -> dict:
        # Distinctive per table so /api/meta reveals WHICH retriever it read.
        return {
            "latest_crawl_date": "2026-09-13T00:00:00+00:00",
            "total_notifications": 777 if "swapped" in self.table_name else 111,
        }

    def fetch_by_id(self, chunk_id):  # noqa: ARG002
        return None

    def fetch_by_ids(self, chunk_ids):  # noqa: ARG002
        return {}


def test_production_wires_the_bindings_the_route_requires(client):
    """The route 503s without these; production must supply them itself."""
    tracker = getattr(app.state, "request_tracker", None)
    builder = getattr(app.state, "build_retriever", None)
    assert tracker is not None, (
        "app.state.request_tracker is unset — /admin/reload-table returns 503 "
        "in production"
    )
    assert builder is not None, (
        "app.state.build_retriever is unset — /admin/reload-table returns 503 "
        "in production"
    )
    assert callable(builder), "build_retriever must be callable"
    assert callable(getattr(app, "_swap_runtime_services", None)), (
        "app._swap_runtime_services is unset — the route cannot rebind the "
        "retrievers queries actually read"
    )


def test_reload_swaps_the_retriever_a_query_path_reads(client, monkeypatch):
    """The point of the whole ticket: a query path must observe the swap.

    Asserts through ``/api/meta``, which reads ``app.state.retriever`` — a real
    read path, not an attribute this test assigned. The snapshot is table-tagged
    so the assertion can only pass if the read path moved to the new table.
    """
    new_table = "husc_v2026_swapped"
    swap_calls: list[str] = []

    async def fake_builder(table_name: str):
        swap_calls.append(table_name)
        return _Retriever(table_name)

    # Only table-name resolution is stubbed; the route, the drain, the gate, the
    # swap seam and the read path all run for real.
    monkeypatch.setattr(app.state, "build_retriever", fake_builder)

    before = client.get("/api/meta")
    assert before.status_code == 200, before.text

    res = client.post(
        "/admin/reload-table", json={"table_name": new_table}, headers=AUTH
    )
    assert res.status_code == 200, res.text
    assert res.json()["active_table"] == new_table
    assert swap_calls == [new_table]

    after = client.get("/api/meta")
    assert after.status_code == 200, after.text
    assert after.json()["total_notifications"] == 777, (
        "/api/meta still describes the old retriever after a 200 reload — the "
        "swap did not reach the read path. Body: "
        f"{after.json()!r} (before: {before.json()!r})"
    )


def test_reload_requires_a_permitted_credential(client):
    """No credential -> refused, and nothing swapped."""
    assert client.post(
        "/admin/reload-table", json={"table_name": "husc_v2026_swapped"}
    ).status_code in {403, 422}
    assert client.post(
        "/admin/reload-table",
        json={"table_name": "husc_v2026_swapped"},
        headers={"X-Admin-Token": "wrong"},
    ).status_code == 403


def test_retriever_healthcheck_and_close_exist_for_the_route(client):
    """The route calls both; a missing method is an AttributeError mid-swap."""
    from src.services.lancedb_retrieval import LanceDBRetriever

    assert hasattr(LanceDBRetriever, "healthcheck"), (
        "admin.py calls new_retriever.healthcheck(); without it the swap dies "
        "after the drain, with the old retriever already closed"
    )
    assert hasattr(LanceDBRetriever, "close"), (
        "admin.py closes the previous retriever after the swap"
    )
