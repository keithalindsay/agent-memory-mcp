"""Regression tests from a chaos-QA battery (2026-07-20).

Each test encodes a CONFIRMED behavioral defect against the DESIGN.md contract
and FAILS on the current code; it will pass once the bug is fixed. Imports use
submodules (``agent_memory_mcp.memory``) rather than the top-level re-export so
the suite is unaffected by the packaging defect exercised in
``test_public_api_reexport_importable`` below.
"""

from __future__ import annotations

import importlib

import pytest

from agent_memory_mcp.memory import Memory
from agent_memory_mcp.vocab import node_id


# --------------------------------------------------------------------------
# Finding 1 — packaging/namespace: the documented public API re-export
# `from agent_memory_mcp import Memory` is unavailable, because the editable
# install resolves the package as a namespace and never executes __init__.py.
# DESIGN.md §10 promises __init__.py "re-exports Memory + public API"; two
# shipped tests (test_recall.py, test_memory_e2e.py) rely on it and error on
# collection under `.venv/bin/pytest`.
# --------------------------------------------------------------------------
def test_public_api_reexport_importable():
    mod = importlib.import_module("agent_memory_mcp")
    # Under the namespace-mode editable install, `mod` has no __file__ and no
    # Memory attribute, so this assertion fails until the package is importable
    # as a real (non-namespace) package with its __init__.py executed.
    assert getattr(mod, "__file__", None) is not None, (
        "agent_memory_mcp imported as a namespace package; __init__.py not executed"
    )
    assert hasattr(mod, "Memory"), "public re-export `from agent_memory_mcp import Memory` missing"


# --------------------------------------------------------------------------
# Finding 2 — non-ASCII / emoji entity names collapse to one node.
# vocab.slug() strips every non-[a-z0-9] char, so any all-non-ASCII name
# (CJK, Arabic, emoji) yields an EMPTY slug and node id "<type>:". Distinct
# entities collide onto one node; the last write wins and recall returns the
# WRONG owner. A knowledge graph must keep distinct entities distinct.
# --------------------------------------------------------------------------
def test_non_ascii_entities_do_not_collide():
    m = Memory(":memory:")
    m.remember("日本語チーム owns the Payments Service.")
    m.remember("мура owns the Billing Service.")

    # Root cause: both names slug to "" so they share id "person:".
    assert node_id("Person", "日本語チーム") != node_id("Person", "мура"), (
        "distinct non-ASCII names produced identical node ids (empty slug collision)"
    )

    payments_owner = m.recall("who owns the Payments Service").model_dump()["answer"]
    billing_owner = m.recall("who owns the Billing Service").model_dump()["answer"]
    assert payments_owner == "日本語チーム", f"wrong owner for Payments Service: {payments_owner!r}"
    assert billing_owner == "мура", f"wrong owner for Billing Service: {billing_owner!r}"


# --------------------------------------------------------------------------
# Finding 3 — forget() over-deletes unrelated facts via unescaped SQL LIKE.
# VectorStore.delete_by_entity runs `text LIKE '%<name>%'`, so forgetting a
# short name deletes every fact whose text merely CONTAINS that name as a
# substring (e.g. forgetting "Ana" destroys facts about "Diana"). DESIGN.md
# §7.1: forget removes "an entity and all facts mentioning IT" — not others.
# --------------------------------------------------------------------------
def test_forget_does_not_over_delete_similar_named_entities():
    m = Memory(":memory:")
    m.remember("Ana manages the Search Team.")
    m.remember("Diana owns the Billing Service.")  # "Diana" contains "ana"

    m.forget("Ana")

    # Diana's fact must survive in the vector store (graph edge is separate).
    hit = m.recall("what does Diana own", method="vector").model_dump()
    assert hit["answer"] is not None, "forget('Ana') wiped Diana's vector facts (over-deletion)"
    assert "Billing" in hit["answer"], f"Diana's fact lost after forgetting Ana: {hit!r}"


# --------------------------------------------------------------------------
# Finding 3b — a name containing a SQL LIKE wildcard ('%' or '_') matches
# EVERY fact, so a single forget destroys the entire vector store.
# --------------------------------------------------------------------------
def test_forget_wildcard_name_does_not_wipe_all_facts():
    m = Memory(":memory:")
    m.remember("Dana owns the Search Service.")
    m.remember("Evan owns the Billing Service.")
    m.remember("% owns the Metrics Service.")

    m.forget("%")

    # Dana's and Evan's facts are unrelated to "%" and must remain.
    dana = m.recall("what does Dana own", method="vector").model_dump()
    evan = m.recall("what does Evan own", method="vector").model_dump()
    assert dana["answer"] is not None and "Search" in dana["answer"], (
        f"forget('%') wiped Dana's facts (LIKE-wildcard over-deletion): {dana!r}"
    )
    assert evan["answer"] is not None and "Billing" in evan["answer"], (
        f"forget('%') wiped Evan's facts (LIKE-wildcard over-deletion): {evan!r}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
