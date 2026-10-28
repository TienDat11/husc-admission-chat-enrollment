"""Behaviour tests for the ns24 gateway contract (config/ns24_models.py + llm_client)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parents[2] / "src")]

from config.ns24_models import PRICE_USD_PER_MTOK  # noqa: E402
from config.request_context import get_request_id, reset_request_id, set_request_id  # noqa: E402
from config.settings import RAGSettings  # noqa: E402
from services import llm_client as lc  # noqa: E402


@pytest.fixture
def ns24_key(monkeypatch):
    monkeypatch.setenv("NS24_API_KEY", "sk-test-ns24")
    lc.reset_llm_clients()
    yield
    lc.reset_llm_clients()


@pytest.fixture
def no_key(monkeypatch):
    monkeypatch.delenv("NS24_API_KEY", raising=False)
    lc.reset_llm_clients()
    yield
    lc.reset_llm_clients()


# (a) whitelist + role fallback wiring

def test_unknown_gen_model_rejected_by_settings(monkeypatch):
    monkeypatch.setenv("NS24_GEN_MODEL", "not-a-model")
    with pytest.raises(ValueError):
        RAGSettings(_env_file=None)


def test_gen_role_builds_primary_and_same_gateway_fallback(ns24_key):
    providers = lc._build_providers(role="gen")
    assert [p.model for p in providers] == ["gpt-6-luna", "claude-haiku-5-5"]
    assert providers[0].fallback_from is None
    assert providers[1].fallback_from == "gpt-6-luna"


# (b) no key => no providers, clear error

def test_no_key_builds_no_providers(no_key):
    assert lc._build_providers() == []


def test_no_key_chat_raises_mentioning_key(no_key):
    client = lc.UnifiedLLMClient()
    with pytest.raises(RuntimeError, match="NS24_API_KEY"):
        asyncio.run(client.chat(user_message="ping", max_tokens=5))


# (c) cost arithmetic

def test_estimate_cost_plain_prompt_only():
    cost, partial = lc.estimate_cost("gemini-3.5-flash-lite", 1_000_000, 0, 0, 0)
    assert cost == pytest.approx(0.01848)
    assert partial is False


def test_estimate_cost_with_cached_input_sums_rates():
    p = PRICE_USD_PER_MTOK["gemini-3.5-flash-lite"]
    # 1M prompt tokens of which 400k are cache reads.
    cost, partial = lc.estimate_cost("gemini-3.5-flash-lite", 1_000_000, 0, 400_000, 0)
    expected = 0.6 * p.input + 0.4 * p.cache_read
    assert cost == pytest.approx(expected)
    assert partial is False


def test_cache_write_without_price_is_partial():
    # gpt-6-luna has no cache_write price.
    _, partial = lc.estimate_cost("gpt-6-luna", 1000, 10, 0, 1000)
    assert partial is True


def test_missing_tokens_returns_none_not_zero():
    assert lc.estimate_cost("gpt-6-luna", None, 10, 0, 0) == (None, False)


# (d) ledger summary

def _usage(**kw):
    return lc._usage_from(
        None if kw.pop("missing", False) else _resp_usage(kw.pop("p", 1000), kw.pop("c", 100)),
        "ns24",
        kw.pop("model", "gemini-3.5-flash-lite"),
        role=kw.pop("role", "gen"),
        attempt=1,
        fallback_from=None,
        latency_ms=5,
        outcome=kw.pop("outcome", "ok"),
    )


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.prompt_tokens_details = None


def _resp_usage(p, c):
    return _Usage(p, c)


def test_summary_one_ok_one_error_marks_partial_and_buckets():
    ledger = lc.TokenLedger()
    ledger.record(_usage(role="gen"))
    ledger.record(_usage(role="router", missing=True, outcome="error"))
    s = ledger.summary()
    assert s["total"]["call_count"] == 2
    assert s["total"]["cost_partial"] is True
    assert set(s["by_role"]) == {"gen", "router"}


def test_summary_unattributed_when_no_request_id():
    ledger = lc.TokenLedger()
    ledger.record(_usage(role="gen"))
    assert "unattributed" in ledger.summary()["by_request"]


# (e) request id contextvar binds onto recorded usage

def test_request_id_flows_into_usage_and_resets():
    token = set_request_id("r1")
    try:
        u = lc._usage_from(_resp_usage(10, 2), "ns24", "gemini-3.5-flash-lite",
                           role="gen", attempt=1, fallback_from=None,
                           latency_ms=1, outcome="ok")
        assert u.request_id == "r1"
    finally:
        reset_request_id(token)
    assert get_request_id() is None


def test_missing_usage_is_null_not_zero():
    u = lc._usage_from(None, "ns24", "gemini-3.5-flash-lite", role="gen",
                       attempt=1, fallback_from=None, latency_ms=1, outcome="ok")
    assert u.prompt_tokens is None
    assert u.completion_tokens is None
    assert u.usage_missing is True
    assert u.cost_usd is None


# ── Ledger: record / snapshot / reset / totals ──


def _priced(role="gen", model="gemini-3.5-flash-lite", p=1000, c=100):
    return lc.TokenUsage(
        provider="ns24", model=model, role=role, prompt_tokens=p,
        completion_tokens=c, cached_input_tokens=0, cache_write_tokens=0,
        outcome="ok", usage_missing=False, cost_usd=0.001, cost_partial=False,
    )


def test_ledger_record_then_snapshot_keeps_order():
    ledger = lc.TokenLedger()
    a, b = _priced(p=1), _priced(p=2)
    ledger.record(a)
    ledger.record(b)
    assert ledger.snapshot() == [a, b]


def test_ledger_reset_empties():
    ledger = lc.TokenLedger()
    ledger.record(_priced())
    ledger.reset()
    assert ledger.snapshot() == []


def test_ledger_totals_aggregate_and_mark_partial():
    ledger = lc.TokenLedger()
    ledger.record(_priced(p=100, c=10))
    ledger.record(_priced(p=200, c=20))
    missing = lc._usage_from(None, "ns24", "gpt-6-luna", role="gen",
                             attempt=1, fallback_from=None, latency_ms=1, outcome="error")
    ledger.record(missing)
    t = ledger.totals()["ns24/gemini-3.5-flash-lite"]
    assert t["prompt_tokens"] == 300
    assert t["completion_tokens"] == 30
    assert t["call_count"] == 2
    assert ledger.totals()["ns24/gpt-6-luna"]["usage_missing_calls"] == 1
    assert ledger.totals()["ns24/gpt-6-luna"]["cost_partial"] is True


def test_aggregate_summary_merges_roles_per_request(ns24_key):
    # Two role clients, one request id spanning both: the per-client ledgers
    # cannot answer "what did this request cost" — the merge must.
    lc.reset_llm_clients()
    try:
        gen = lc.get_llm_client(role="gen")
        router = lc.get_llm_client(role="router")
        token = set_request_id("req-merge")
        try:
            gen.token_ledger.record(lc._usage_from(_resp_usage(100, 10), "ns24", "gemini-3.5-flash-lite",
                                                   role="gen", attempt=1, fallback_from=None,
                                                   latency_ms=1, outcome="ok"))
            router.token_ledger.record(lc._usage_from(_resp_usage(50, 5), "ns24", "gemini-3.5-flash-lite",
                                                     role="router", attempt=1, fallback_from=None,
                                                     latency_ms=1, outcome="ok"))
        finally:
            reset_request_id(token)
        # A call with no request id must stay visibly unattributed.
        gen.token_ledger.record(_priced(role="gen", p=7, c=1))

        agg = lc.aggregate_ledger_summary()
        assert set(agg["by_request"]) == {"req-merge", "unattributed"}
        assert agg["by_request"]["req-merge"]["call_count"] == 2
        assert agg["by_request"]["req-merge"]["prompt_tokens"] == 150
        assert agg["total"]["call_count"] == 3
        assert agg["total"]["prompt_tokens"] == 157
        assert set(agg["by_role"]) == {"gen", "router"}
    finally:
        lc.reset_llm_clients()


def test_aggregate_summary_reads_only_existing_clients(ns24_key):
    # Reading the aggregate must never build a client as a side effect.
    lc.reset_llm_clients()
    try:
        assert lc._clients == {}
        agg = lc.aggregate_ledger_summary()
        assert agg["total"]["call_count"] == 0
        assert lc._clients == {}
    finally:
        lc.reset_llm_clients()
