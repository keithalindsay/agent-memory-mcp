"""MCP tool adapter tests — call the FastMCP tool callables directly (§11 step 11)."""

import pytest

from agent_memory_mcp import server


@pytest.fixture(autouse=True)
def fresh_memory():
    server.configure(db=":memory:", extractor="rules", embedder="hash")
    yield
    server.configure(db=":memory:")


def test_remember_returns_json_dict():
    out = server.remember("Dana is an Engineer. Dana is managed by Evan.")
    assert isinstance(out, dict)
    assert out["relations_added"] >= 2
    assert out["stored_raw"] >= 1
    assert isinstance(out["note"], str)
    assert {"src", "rel", "dst"} <= set(out["learned"][0].keys())


def test_remember_never_throws_on_garbage():
    out = server.remember("asdfjkl qwerty zxcvbnm.")
    assert isinstance(out, dict)
    assert out["relations_added"] == 0
    assert out["stored_raw"] >= 1
    assert "raw text" in out["note"].lower()


def test_recall_multi_hop_graph():
    server.remember("Dana is managed by Evan.")
    server.remember("Evan is managed by Farah.")
    out = server.recall("Who does Dana's manager report to?")
    assert out["answer"] == "Farah"
    assert out["method"] == "graph"


def test_recall_miss_returns_none_not_error():
    server.remember("Dana is managed by Evan.")
    out = server.recall("What is the capital of Peru?")
    assert out["answer"] is None
    assert out["method"] == "none"


def test_compare_returns_both_backends():
    server.remember("Dana is managed by Evan.")
    server.remember("Evan is managed by Farah.")
    out = server.compare("Who does Dana's manager report to?")
    assert set(out.keys()) == {"query", "graph", "vector", "agree", "note"}
    assert out["graph"]["answer"] == "Farah"
    assert "answer" in out["vector"]
    assert isinstance(out["agree"], bool)


def test_neighbors():
    server.remember("Dana is managed by Evan.")
    out = server.neighbors("Dana")
    assert out["entity"] == "Dana"
    assert any(o["rel"] == "MANAGED_BY" and o["dst"] == "Evan" for o in out["out"])


def test_entities_and_filter():
    server.remember("Dana is managed by Evan.")
    allents = server.entities()
    assert allents["count"] >= 2
    filtered = server.entities("dana")
    assert filtered["count"] == 1


def test_forget_removes():
    server.remember("Dana is managed by Evan.")
    out = server.forget("Dana")
    assert out["removed_entity"] == "Dana"
    assert out["edges_removed"] >= 1
    assert server.entities("dana")["count"] == 0


def test_stats_counts():
    server.remember("Dana is managed by Evan.")
    server.remember("Evan is managed by Farah.")
    out = server.stats()
    assert out["entities"] == 3
    assert out["edges"] == 2
    assert out["relations"].get("MANAGED_BY") == 2
