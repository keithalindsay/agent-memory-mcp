"""The recall pipeline (DESIGN.md §6 recall.py, §5 flow).

resolve_entities -> infer_relation -> graph_answer (traverse) -> vector fallback.
"""

from __future__ import annotations

import re
from typing import Optional

from .embedders import Embedder
from .graph_store import GraphStore
from .models import Entity, RecallResult
from .vector_store import VectorStore
from .vocab import QUERY_TRIGGERS, fact_sentence

_QUERY_TRIGGERS_SORTED = sorted(QUERY_TRIGGERS, key=lambda t: len(t[0]), reverse=True)


def resolve_entities(query: str, graph: GraphStore) -> list[tuple[Entity, tuple[int, int]]]:
    """Find known entities named in the query. Returns (entity, char-span), longest first."""
    low = query.lower()
    nodes = graph.all_nodes()
    nodes.sort(key=lambda n: len(n.name), reverse=True)
    consumed = [False] * len(low)
    out: list[tuple[Entity, tuple[int, int]]] = []
    for node in nodes:
        name = node.name.lower()
        if not name:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", low):
            if any(consumed[m.start() : m.end()]):
                continue
            for i in range(m.start(), m.end()):
                consumed[i] = True
            out.append((node, (m.start(), m.end())))
            break
    out.sort(key=lambda t: len(t[0].name), reverse=True)
    return out


def infer_relations(query: str, mask_spans: Optional[list[tuple[int, int]]] = None) -> list[str]:
    """Infer relation(s) from query keywords, in order of appearance, with repeats.

    Character spans in ``mask_spans`` (e.g. resolved entity names) are ignored so
    that a word like "Team" inside an entity name does not fire MEMBER_OF.
    """
    low = query.lower()
    masked = list(low)
    for start, end in mask_spans or []:
        for i in range(start, min(end, len(masked))):
            masked[i] = "\x00"
    text = "".join(masked)
    consumed = [False] * len(text)
    hits: list[tuple[int, str]] = []
    for phrase, rel in _QUERY_TRIGGERS_SORTED:
        for m in re.finditer(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text):
            if any(consumed[m.start() : m.end()]):
                continue
            for i in range(m.start(), m.end()):
                consumed[i] = True
            hits.append((m.start(), rel))
    hits.sort(key=lambda t: t[0])
    return [rel for _pos, rel in hits]


def expected_answer_type(query: str) -> Optional[str]:
    low = query.lower()
    for kw, typ in [
        ("project", "Project"),
        ("service", "Service"),
        ("team", "Team"),
        ("database", "Service"),
    ]:
        if f"which {kw}" in low or f"what {kw}" in low:
            return typ
    if re.search(r"\bwho\b|\bwhom\b", low):
        return "Person"
    return None


def graph_answer(query: str, graph: GraphStore) -> Optional[RecallResult]:
    """Traverse the graph to answer the query; returns a RecallResult or None."""
    resolved = resolve_entities(query, graph)
    if not resolved:
        return None
    spans = [span for _e, span in resolved]
    rels = infer_relations(query, mask_spans=spans)
    if not rels:
        return None
    rel_set = set(rels)
    depth = min(len(rels), 4)
    want_type = expected_answer_type(query)
    exclude = {e.id for e, _ in resolved}

    for anchor, _span in resolved:
        result = _bfs_answer(graph, anchor.id, rel_set, depth, want_type, exclude)
        if result is not None:
            return result
    return None


def _bfs_answer(
    graph: GraphStore,
    start: str,
    rel_set: set[str],
    target_depth: int,
    want_type: Optional[str],
    exclude: set[str],
) -> Optional[RecallResult]:
    """Layered traversal along any relation in rel_set (either direction).

    A path step always shows the true edge orientation ("src --REL--> dst"),
    regardless of which way we walked it.
    """
    # frontier states: (node_id, edges) where edges = [(src_id, rel, dst_id), ...]
    frontier: list[tuple[str, list[tuple[str, str, str]]]] = [(start, [])]
    visited_nodes = {start}
    candidates_by_depth: dict[int, list[tuple[str, list[tuple[str, str, str]]]]] = {}

    for depth in range(1, target_depth + 1):
        next_frontier: list[tuple[str, list[tuple[str, str, str]]]] = []
        for node, edges in frontier:
            for rel in rel_set:
                for _rel, other in graph.neighbors(node, rel=rel, direction="out"):
                    if other in visited_nodes:
                        continue
                    next_frontier.append((other, edges + [(node, rel, other)]))
                for _rel, other in graph.neighbors(node, rel=rel, direction="in"):
                    if other in visited_nodes:
                        continue
                    next_frontier.append((other, edges + [(other, rel, node)]))
        for node, _edges in next_frontier:
            visited_nodes.add(node)
        candidates_by_depth[depth] = next_frontier
        frontier = next_frontier
        if not frontier:
            break

    # Prefer candidates at the deepest reached level (closest to target).
    for depth in sorted(candidates_by_depth.keys(), reverse=True):
        cands = [c for c in candidates_by_depth[depth] if c[0] not in exclude]
        if not cands:
            continue
        chosen = _pick(graph, cands, want_type)
        if chosen is None:
            continue
        node, edges = chosen
        node_obj = graph.get_node(node)
        answer = node_obj.name if node_obj else node
        path, support = _render(graph, edges)
        return RecallResult(answer=answer, method="graph", support=support, path=path)
    return None


def _pick(graph, cands, want_type):
    if want_type:
        typed = [
            c
            for c in cands
            if (graph.get_node(c[0]) or None) and graph.get_node(c[0]).type == want_type
        ]
        if typed:
            return typed[0]
    return cands[0]


def _render(graph: GraphStore, edges: list[tuple[str, str, str]]):
    def nm(nid: str) -> str:
        n = graph.get_node(nid)
        return n.name if n else nid

    path = [f"{nm(s)} --{r}--> {nm(d)}" for (s, r, d) in edges]
    support = [fact_sentence(r, nm(s), nm(d)) for (s, r, d) in edges]
    return path, support


# Below this cosine, a vector hit is treated as "no confident match" (a miss)
# rather than a weak, off-topic answer.
VECTOR_MIN_SCORE = 0.15


def vector_answer(
    query: str,
    vector: VectorStore,
    embedder: Embedder,
    k: int = 4,
    min_score: float = VECTOR_MIN_SCORE,
) -> Optional[RecallResult]:
    """Fallback: embed the query and return the best-matching stored fact."""
    hits = vector.search(embedder.embed(query), k=k)
    if not hits or hits[0].score < min_score:
        return None
    best = hits[0]
    # de-duplicate support text, keep order
    support: list[str] = []
    for h in hits:
        if h.text not in support:
            support.append(h.text)
    return RecallResult(answer=best.text, method="vector", support=support, path=[])


def phrase(result: RecallResult, mode: str = "rules") -> RecallResult:
    """Phrase the answer. Rules mode is a pass-through; llm phrasing is optional."""
    return result
