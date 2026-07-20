"""Tests for embedders, vector store, and graph store (DESIGN.md §12)."""

import numpy as np

from agent_memory_mcp.embedders import HashEmbedder, get_embedder
from agent_memory_mcp.graph_store import GraphStore
from agent_memory_mcp.vector_store import VectorStore

# --- Embedder ---------------------------------------------------------------


def test_hash_embedder_is_deterministic():
    e = HashEmbedder(dim=256)
    v1 = e.embed("Dana is managed by Evan")
    v2 = e.embed("Dana is managed by Evan")
    assert v1.shape == (256,)
    assert np.allclose(v1, v2)


def test_hash_embedder_self_cosine_is_one():
    e = HashEmbedder(dim=256)
    v = e.embed("the Search Team owns the Index Service")
    cos = float(np.dot(v, v))  # already L2-normalized
    assert abs(cos - 1.0) < 1e-6


def test_hash_embedder_similar_text_closer_than_unrelated():
    e = HashEmbedder(dim=256)
    q = e.embed("index service")
    close = e.embed("the Search Team owns the Index Service")
    far = e.embed("bananas grow in tropical climates")
    assert float(np.dot(q, close)) > float(np.dot(q, far))


def test_get_embedder_factory():
    e = get_embedder("hash", dim=128)
    assert e.embed("x").shape == (128,)


# --- VectorStore ------------------------------------------------------------


def test_vector_store_roundtrip_and_nearest_hit():
    e = HashEmbedder(dim=256)
    vs = VectorStore(":memory:", dim=256)
    facts = [
        "The Search Team owns the Index Service.",
        "Atlas uses the Index Service.",
        "Bananas grow in tropical climates.",
    ]
    for text in facts:
        vs.add(fact_id=None, text=text, src=None, rel=None, dst=None, vector=e.embed(text))
    hits = vs.search(e.embed("who owns the index service"), k=2)
    assert len(hits) == 2
    assert "Index Service" in hits[0].text
    assert 0.0 <= hits[0].score <= 1.0001


def test_vector_store_delete_by_entity():
    e = HashEmbedder(dim=256)
    vs = VectorStore(":memory:", dim=256)
    vs.add(
        None,
        "Dana is managed by Evan.",
        "person:dana",
        "MANAGED_BY",
        "person:evan",
        e.embed("Dana is managed by Evan."),
    )
    vs.add(None, "Bananas grow in tropical climates.", None, None, None, e.embed("bananas"))
    removed = vs.delete_by_entity("person:dana")
    assert removed == 1
    hits = vs.search(e.embed("Dana"), k=5)
    assert all("Dana" not in h.text for h in hits)


# --- GraphStore -------------------------------------------------------------


def test_graph_store_node_edge_roundtrip():
    g = GraphStore(":memory:")
    g.add_node("person:dana", "Person", "Dana")
    g.add_node("person:evan", "Person", "Evan")
    g.add_edge("person:dana", "MANAGED_BY", "person:evan")
    node = g.get_node("person:dana")
    assert node is not None and node.name == "Dana"
    # idempotent upsert
    g.add_node("person:dana", "Person", "Dana")
    g.add_edge("person:dana", "MANAGED_BY", "person:evan")
    out = g.neighbors("person:dana", direction="out")
    assert ("MANAGED_BY", "person:evan") in [(r, d) for (r, d) in out]


def test_graph_store_find_nodes_case_insensitive():
    g = GraphStore(":memory:")
    g.add_node("team:search-team", "Team", "Search Team")
    found = g.find_nodes("search team")
    assert any(n.id == "team:search-team" for n in found)


def test_graph_store_remove_entity():
    g = GraphStore(":memory:")
    g.add_node("person:dana", "Person", "Dana")
    g.add_node("person:evan", "Person", "Evan")
    g.add_edge("person:dana", "MANAGED_BY", "person:evan")
    edges_removed = g.remove_entity("person:dana")
    assert edges_removed == 1
    assert g.get_node("person:dana") is None
