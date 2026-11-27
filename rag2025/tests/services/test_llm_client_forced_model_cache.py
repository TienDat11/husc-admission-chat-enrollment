"""Forced-model clients are cached per (role, model) and closed on shutdown."""
import asyncio

import pytest

from services import llm_client as lc


class FakeClient:
    instances: list = []

    def __init__(self, force_model=None, role="default"):
        self.force_model = force_model
        self.role = role
        self.closed = False
        FakeClient.instances.append(self)

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    monkeypatch.setattr(lc, "UnifiedLLMClient", FakeClient)
    lc.reset_llm_clients()
    FakeClient.instances.clear()
    yield
    lc.reset_llm_clients()


def test_forced_model_client_reused_for_same_role_and_model():
    a = lc.get_llm_client(force_model="m1", role="gen")
    b = lc.get_llm_client(force_model="m1", role="gen")
    assert a is b
    assert len(FakeClient.instances) == 1


def test_distinct_forced_models_get_distinct_clients():
    a = lc.get_llm_client(force_model="m1", role="gen")
    b = lc.get_llm_client(force_model="m2", role="gen")
    assert a is not b


def test_forced_client_does_not_replace_role_singleton():
    shared = lc.get_llm_client(role="gen")
    forced = lc.get_llm_client(force_model="m1", role="gen")
    assert shared is not forced
    assert lc.get_llm_client(role="gen") is shared


def test_close_llm_clients_closes_all_and_empties_registry():
    forced = lc.get_llm_client(force_model="m1", role="gen")
    shared = lc.get_llm_client(role="gen")
    asyncio.run(lc.close_llm_clients())
    assert forced.closed and shared.closed
    assert lc._clients == {}
