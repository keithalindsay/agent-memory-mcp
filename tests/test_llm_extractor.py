"""LlmExtractor tests with the Anthropic SDK MOCKED (no network) — DESIGN.md §11 step 12."""

import sys
import types
from types import SimpleNamespace

import pytest

from agent_memory_mcp.extractors import LlmExtractor


class _FakeMessages:
    def __init__(self, payload_text: str):
        self._payload_text = payload_text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self._payload_text)])


class _FakeClient:
    def __init__(self, payload_text: str):
        self.messages = _FakeMessages(payload_text)


@pytest.fixture()
def fake_anthropic(monkeypatch):
    """Install a fake `anthropic` module so no real network call is possible."""
    payload = {"text": '{"relations": [{"src": "Dana", "rel": "MANAGED_BY", "dst": "Evan"}]}'}
    created = {}

    def _factory(api_key=None):
        client = _FakeClient(payload["text"])
        created["client"] = client
        created["api_key"] = api_key
        return client

    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = _factory
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    return payload, created


def test_llm_extractor_parses_mocked_response(fake_anthropic):
    _payload, created = fake_anthropic
    ex = LlmExtractor()
    statements = ex.extract("Dana is managed by Evan.")
    triples = {(r.src, r.rel, r.dst) for st in statements for r in st.relations}
    assert ("person:dana", "MANAGED_BY", "person:evan") in triples
    # confirms the (mocked) SDK was constructed with the env key and never hit the network
    assert created["api_key"] == "test-key-not-real"


def test_llm_extractor_uses_model_from_config(fake_anthropic, monkeypatch):
    monkeypatch.setenv("AGENT_MEMORY_MODEL", "claude-test-model-xyz")
    _payload, created = fake_anthropic
    ex = LlmExtractor()
    ex.extract("Dana is managed by Evan.")
    assert created["client"].messages.calls[0]["model"] == "claude-test-model-xyz"


def test_llm_extractor_default_model_is_haiku(fake_anthropic, monkeypatch):
    monkeypatch.delenv("AGENT_MEMORY_MODEL", raising=False)
    _payload, created = fake_anthropic
    ex = LlmExtractor()
    ex.extract("Dana is managed by Evan.")
    assert created["client"].messages.calls[0]["model"] == "claude-haiku-4-5"


def test_llm_extractor_degrades_on_bad_json(fake_anthropic):
    payload, _created = fake_anthropic
    payload["text"] = "this is not json at all"
    ex = LlmExtractor()
    # never throws: returns raw statements with no relations
    statements = ex.extract("Some unparseable prose.")
    assert statements
    assert all(not st.relations for st in statements)


def test_llm_extractor_missing_key_raises(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    fake_mod = types.ModuleType("anthropic")
    fake_mod.Anthropic = lambda api_key=None: _FakeClient("{}")
    monkeypatch.setitem(sys.modules, "anthropic", fake_mod)
    ex = LlmExtractor()
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ex.extract("Dana is managed by Evan.")
