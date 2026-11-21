"""``table_tsv`` on the exceptions payload — no database needed.

The route reads ``document_blocks`` through ``make_session_factory``; a fake
factory stands in for it so the route body runs on a synthetic payload.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.routers import admin_exceptions  # noqa: E402
from src.routers.admin_exceptions import ExceptionsListResponse  # noqa: E402


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _Session:
    def __init__(self, rows):
        self._rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, *_args, **_kwargs):
        return _Result(self._rows)


def _block(block_id: str, table_tsv, block_type: str = "table") -> dict:
    return {
        "block_id": block_id,
        "document_id": "uploads/abc.pdf",
        "page": 1,
        "ordinal": 0,
        "block_type": block_type,
        "text": "Bảng điểm chuẩn",
        "table_tsv": table_tsv,
        "warnings": [],
        "version": 1,
    }


def _run(monkeypatch, block_rows: list[dict]) -> dict:
    """Call the route with synthetic block rows; no unanchored records."""
    sessions = iter([block_rows, []])
    monkeypatch.setattr(
        admin_exceptions,
        "make_session_factory",
        lambda _engine: lambda: _Session(next(sessions)),
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db_engine=object())))
    response = asyncio.run(
        admin_exceptions.list_exceptions(
            document_id="uploads/abc.pdf", request=request, identity=None
        )
    )
    return response.model_dump()


def test_table_tsv_is_returned_per_block(monkeypatch) -> None:
    tsv = "Mã ngành\tTên ngành\n7480201\tCông nghệ thông tin"
    payload = _run(monkeypatch, [_block("b1", tsv)])

    assert payload["blocks"][0]["table_tsv"] == tsv


def test_table_tsv_is_null_when_unavailable(monkeypatch) -> None:
    payload = _run(
        monkeypatch,
        [
            _block("b_null", None, block_type="text"),
            _block("b_empty", "", block_type="text"),
        ],
    )

    assert [b["table_tsv"] for b in payload["blocks"]] == [None, None]


def test_existing_block_fields_are_unchanged(monkeypatch) -> None:
    payload = _run(monkeypatch, [_block("b1", "a\tb")])

    block = payload["blocks"][0]
    assert set(block) == {
        "block_id",
        "document_id",
        "page",
        "ordinal",
        "block_type",
        "text",
        "table_tsv",
        "state",
        "warnings",
        "missing_pypdf_rows",
        "coordinate_key",
        "version",
    }
    assert block["state"] == "open"
    assert payload["unanchored"] == []


def test_openapi_schema_declares_nullable_table_tsv() -> None:
    schema = ExceptionsListResponse.model_json_schema()
    prop = schema["$defs"]["BlockView"]["properties"]["table_tsv"]

    assert prop == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "default": None,
        "title": "Table Tsv",
    }
