"""Recall pipeline tests (DESIGN.md §11 step 9, §12)."""

import pytest

from agent_memory_mcp import Memory

SEED = [
    "Dana is an Engineer.",
    "Dana is managed by Evan.",
    "Evan is managed by Farah.",
    "Dana is a member of the Search Team.",
    "The Search Team owns the Index Service.",
    "Atlas uses the Index Service.",
]


@pytest.fixture()
def mem() -> Memory:
    m = Memory(":memory:")
    for line in SEED:
        m.remember(line)
    return m


def test_single_hop_role(mem):
    r = mem.recall("What is Dana's role?", method="graph")
    assert r.method == "graph"
    assert "engineer" in (r.answer or "").lower()


def test_single_hop_manager(mem):
    r = mem.recall("Who is Dana's manager?", method="graph")
    assert r.method == "graph"
    assert r.answer == "Evan"
    assert r.path == ["Dana --MANAGED_BY--> Evan"]


def test_multi_hop_reports_to_via_graph(mem):
    r = mem.recall("Who does Dana's manager report to?", method="graph")
    assert r.method == "graph"
    assert r.answer == "Farah"
    assert r.path == ["Dana --MANAGED_BY--> Evan", "Evan --MANAGED_BY--> Farah"]
    assert any("Evan" in s for s in r.support)


def test_aggregation_via_graph(mem):
    r = mem.recall("Which project uses a service owned by the Search Team?", method="graph")
    assert r.method == "graph"
    assert r.answer == "Atlas"


def test_vector_fallback_when_no_relation(mem):
    r = mem.recall("Tell me about the Index Service", method="auto")
    assert r.method == "vector"
    assert "Index Service" in " ".join(r.support)


def test_pinned_vector_method(mem):
    r = mem.recall("Who does Dana's manager report to?", method="vector")
    assert r.method == "vector"


def test_miss_returns_none(mem):
    r = mem.recall("Who is the president of France?", method="auto")
    assert r.method == "none"
    assert r.answer is None


def test_graph_pinned_miss_is_none(mem):
    r = mem.recall("What color is the sky?", method="graph")
    assert r.method == "none"
    assert r.answer is None


def test_remember_is_idempotent(mem):
    before = mem.stats()
    for line in SEED:
        mem.remember(line)
    after = mem.stats()
    assert before["entities"] == after["entities"]
    assert before["edges"] == after["edges"]


def test_remember_learned_relations():
    m = Memory(":memory:")
    res = m.remember("Dana is managed by Evan.")
    assert res.relations_added == 1
    assert res.learned[0].rel == "MANAGED_BY"
    assert res.stored_raw >= 1
