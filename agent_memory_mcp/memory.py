"""The Memory façade (DESIGN.md §5, §6) — one object the server and CLI share."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

from . import recall as recall_mod
from .embedders import Embedder, get_embedder
from .extractors import Extractor, get_extractor
from .graph_store import GraphStore
from .models import RecallResult, RememberResult
from .vector_store import VectorStore, fact_id_for
from .vocab import fact_sentence


def _norm_answer(text: Optional[str]) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


class Memory:
    """Ties the graph store, vector store, extractor, and embedder together."""

    def __init__(
        self,
        db: str = ":memory:",
        embedder: Optional[Embedder] = None,
        extractor: Optional[Extractor] = None,
        dim: int = 256,
        k: int = 4,
    ) -> None:
        self.dim = dim
        self.k = k
        self.embedder = embedder or get_embedder("hash", dim=dim)
        self.extractor = extractor or get_extractor("rules")
        resolved = db
        if db not in (":memory:", ""):
            resolved = str(Path(db).expanduser())
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        # one shared connection => one db file holding all four tables
        self._conn = sqlite3.connect(resolved)
        self.graph = GraphStore(resolved, conn=self._conn)
        self.vector = VectorStore(resolved, dim=self.dim, conn=self._conn)

    # --- write ---
    def remember(self, text: str) -> RememberResult:
        """Split -> extract -> upsert nodes/edges -> always embed raw statement."""
        result = RememberResult()
        statements = self.extractor.extract(text)
        for st in statements:
            name_by_id = {e.id: e.name for e in st.entities}
            for ent in st.entities:
                if self.graph.get_node(ent.id) is None:
                    result.entities_added += 1
                self.graph.add_node(ent.id, ent.type, ent.name, ent.attrs)
            for rel in st.relations:
                if not self.graph.has_edge(rel.src, rel.rel, rel.dst):
                    result.relations_added += 1
                    result.learned.append(rel)
                self.graph.add_edge(rel.src, rel.rel, rel.dst)
                # store the templated fact sentence (tagged) for vector recall
                src_name = name_by_id.get(rel.src, rel.src)
                dst_name = name_by_id.get(rel.dst, rel.dst)
                sent = fact_sentence(rel.rel, src_name, dst_name)
                self.vector.add(
                    fact_id_for(sent), sent, rel.src, rel.rel, rel.dst, self.embedder.embed(sent)
                )
            # always embed the raw statement (even with zero relations)
            if st.text.strip():
                self.vector.add(
                    fact_id_for(st.text), st.text, None, None, None, self.embedder.embed(st.text)
                )
                result.stored_raw += 1

        n_stmts = len([s for s in statements if s.text.strip()])
        if result.relations_added:
            note = (
                f"Learned {result.relations_added} relationship(s) across {n_stmts} statement(s)."
            )
        else:
            note = (
                f"Stored {result.stored_raw} statement(s) as raw text for vector recall; "
                "no structured relationships were extracted."
            )
        unstructured = [s.text for s in statements if s.text.strip() and not s.relations]
        if unstructured and result.relations_added:
            note += f" {len(unstructured)} statement(s) kept as raw text only."
        result.note = note
        return result

    # --- read / recall ---
    def recall(self, query: str, method: str = "auto", k: Optional[int] = None) -> RecallResult:
        k = k or self.k
        method = (method or "auto").lower()
        if method in ("auto", "graph"):
            res = recall_mod.graph_answer(query, self.graph)
            if res is not None:
                return recall_mod.phrase(res)
            if method == "graph":
                return RecallResult(answer=None, method="none")
        if method in ("auto", "vector"):
            res = recall_mod.vector_answer(query, self.vector, self.embedder, k=k)
            if res is not None:
                return recall_mod.phrase(res)
            if method == "vector":
                return RecallResult(answer=None, method="none")
        return RecallResult(answer=None, method="none")

    def compare(self, query: str, k: Optional[int] = None) -> dict:
        """THE benchmarking tool: answer the same query with both backends."""
        g = self.recall(query, method="graph", k=k)
        v = self.recall(query, method="vector", k=k)
        agree = (
            bool(g.answer) and bool(v.answer) and _norm_answer(g.answer) == _norm_answer(v.answer)
        )
        if g.answer and len(g.path) >= 2:
            note = (
                f"Graph traversed {len(g.path)} hops to reach the answer; "
                "vector similarity stopped at the single closest chunk."
            )
        elif g.answer and not v.answer:
            note = "Graph answered by traversal; vector found no confident match."
        elif not g.answer and v.answer:
            note = "No graph path resolved; only vector similarity returned a result."
        else:
            note = "Both backends returned a result."
        return {
            "query": query,
            "graph": {"answer": g.answer, "support": g.support, "path": g.path},
            "vector": {"answer": v.answer, "support": v.support},
            "agree": agree,
            "note": note,
        }

    # --- inspection ---
    def entities(self, filter: str = "") -> dict:
        nodes = self.graph.all_nodes(filter)
        return {
            "entities": [{"id": n.id, "type": n.type, "name": n.name} for n in nodes],
            "count": len(nodes),
        }

    def neighbors(self, entity: str) -> dict:
        matches = self.graph.find_nodes(entity)
        if not matches:
            return {"entity": None, "out": [], "in": []}
        node = matches[0]

        def nm(nid: str) -> str:
            n = self.graph.get_node(nid)
            return n.name if n else nid

        out = [{"rel": r, "dst": nm(o)} for r, o in self.graph.neighbors(node.id, direction="out")]
        inc = [{"rel": r, "src": nm(o)} for r, o in self.graph.neighbors(node.id, direction="in")]
        return {"entity": node.name, "out": out, "in": inc}

    def forget(self, subject: str) -> dict:
        matches = self.graph.find_nodes(subject)
        if not matches:
            return {"removed_entity": None, "edges_removed": 0, "facts_removed": 0}
        node = matches[0]
        edges_removed = self.graph.remove_entity(node.id)
        facts_removed = self.vector.delete_by_entity(node.id, node.name)
        return {
            "removed_entity": node.name,
            "edges_removed": edges_removed,
            "facts_removed": facts_removed,
        }

    def stats(self) -> dict:
        return {
            "entities": self.graph.node_count(),
            "edges": self.graph.edge_count(),
            "facts": self.vector.count(),
            "relations": self.graph.relation_counts(),
        }
