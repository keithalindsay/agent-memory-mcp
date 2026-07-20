"""Traversal tests on a hand-built seed graph (DESIGN.md §11 step 6, §12)."""

import pytest

from agent_memory_mcp.graph_store import GraphStore


@pytest.fixture()
def graph() -> GraphStore:
    g = GraphStore(":memory:")
    for nid, typ, name in [
        ("person:dana", "Person", "Dana"),
        ("person:evan", "Person", "Evan"),
        ("person:farah", "Person", "Farah"),
        ("team:search-team", "Team", "Search Team"),
        ("service:index-service", "Service", "Index Service"),
        ("project:atlas", "Project", "Atlas"),
    ]:
        g.add_node(nid, typ, name)
    g.add_edge("person:dana", "MANAGED_BY", "person:evan")
    g.add_edge("person:evan", "MANAGED_BY", "person:farah")
    g.add_edge("person:dana", "MEMBER_OF", "team:search-team")
    g.add_edge("team:search-team", "OWNS", "service:index-service")
    g.add_edge("project:atlas", "USES", "service:index-service")
    return g


def test_multi_hop_managed_by_reaches_farah(graph):
    paths = graph.traverse("person:dana", "MANAGED_BY", hops=2)
    assert paths, "expected at least one path"
    # the longest path should terminate at Farah
    longest = max(paths, key=lambda p: len(p.steps))
    assert longest.steps[-1] == "Evan --MANAGED_BY--> Farah"
    assert longest.steps[0] == "Dana --MANAGED_BY--> Evan"


def test_single_hop_managed_by(graph):
    paths = graph.traverse("person:dana", "MANAGED_BY", hops=1)
    assert paths[0].steps[-1] == "Dana --MANAGED_BY--> Evan"


def test_aggregation_owns_then_used_by_resolves_atlas(graph):
    # Search Team --OWNS--> Index Service ; who USES that service (incoming) -> Atlas
    owned = graph.neighbors("team:search-team", rel="OWNS", direction="out")
    assert owned == [("OWNS", "service:index-service")]
    service = owned[0][1]
    users = graph.neighbors(service, rel="USES", direction="in")
    assert ("USES", "project:atlas") in users
    assert graph.get_node("project:atlas").name == "Atlas"
