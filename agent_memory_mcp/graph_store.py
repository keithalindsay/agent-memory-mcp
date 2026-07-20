"""Knowledge-graph store (nodes + edges) over SQLite (DESIGN.md §8.2)."""

from __future__ import annotations

import json
import sqlite3
from typing import Optional

from .models import Entity, Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
  id    TEXT PRIMARY KEY,
  type  TEXT NOT NULL,
  name  TEXT NOT NULL,
  attrs TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS edges (
  src TEXT NOT NULL,
  rel TEXT NOT NULL,
  dst TEXT NOT NULL,
  PRIMARY KEY (src, rel, dst)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
"""


class GraphStore:
    """nodes + edges in SQLite with idempotent upserts and simple traversal."""

    def __init__(self, db_path: str, conn: Optional[sqlite3.Connection] = None) -> None:
        self._conn = conn or sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    # --- writes ---
    def add_node(self, node_id: str, type_: str, name: str, attrs: Optional[dict] = None) -> None:
        """Insert or update a node (idempotent upsert by id)."""
        self._conn.execute(
            "INSERT INTO nodes (id, type, name, attrs) VALUES (?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET type=excluded.type, name=excluded.name",
            (node_id, type_, name, json.dumps(attrs or {})),
        )
        self._conn.commit()

    def add_edge(self, src: str, rel: str, dst: str) -> None:
        """Insert an edge; idempotent via the composite primary key."""
        self._conn.execute(
            "INSERT OR IGNORE INTO edges (src, rel, dst) VALUES (?,?,?)", (src, rel, dst)
        )
        self._conn.commit()

    # --- reads ---
    def get_node(self, node_id: str) -> Optional[Entity]:
        row = self._conn.execute(
            "SELECT id, type, name, attrs FROM nodes WHERE id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return Entity(id=row["id"], type=row["type"], name=row["name"], attrs=json.loads(row["attrs"]))

    def find_nodes(self, name: str) -> list[Entity]:
        """Case-insensitive match: exact-name first, then substring, longest first."""
        needle = name.strip().lower()
        if not needle:
            return []
        rows = self._conn.execute("SELECT id, type, name, attrs FROM nodes").fetchall()
        exact, partial = [], []
        for r in rows:
            nm = r["name"].lower()
            ent = Entity(id=r["id"], type=r["type"], name=r["name"], attrs=json.loads(r["attrs"]))
            if nm == needle or r["id"].lower() == needle:
                exact.append(ent)
            elif needle in nm or nm in needle:
                partial.append(ent)
        partial.sort(key=lambda e: len(e.name), reverse=True)
        return exact + partial

    def all_nodes(self, filter_: str = "") -> list[Entity]:
        rows = self._conn.execute(
            "SELECT id, type, name, attrs FROM nodes ORDER BY name"
        ).fetchall()
        out = []
        f = filter_.strip().lower()
        for r in rows:
            if f and f not in r["name"].lower():
                continue
            out.append(Entity(id=r["id"], type=r["type"], name=r["name"], attrs=json.loads(r["attrs"])))
        return out

    def neighbors(
        self, node_id: str, rel: Optional[str] = None, direction: str = "out"
    ) -> list[tuple[str, str]]:
        """Return (rel, other_id) pairs. direction: 'out' | 'in'."""
        if direction == "out":
            sql = "SELECT rel, dst AS other FROM edges WHERE src = ?"
        elif direction == "in":
            sql = "SELECT rel, src AS other FROM edges WHERE dst = ?"
        else:
            raise ValueError("direction must be 'out' or 'in'")
        params: list = [node_id]
        if rel is not None:
            sql += " AND rel = ?"
            params.append(rel)
        rows = self._conn.execute(sql, params).fetchall()
        return [(r["rel"], r["other"]) for r in rows]

    def traverse(self, start: str, relation: str, hops: int = 2) -> list[Path]:
        """Follow ``relation`` outward from ``start`` up to ``hops`` steps.

        Returns one Path per reachable terminal node, with human-readable steps
        like ``"Dana --MANAGED_BY--> Evan"``. Cycles are guarded against.
        """
        results: list[Path] = []

        def name_of(nid: str) -> str:
            n = self.get_node(nid)
            return n.name if n else nid

        def walk(current: str, steps: list[str], visited: set[str], depth: int) -> None:
            outs = self.neighbors(current, rel=relation, direction="out")
            if not outs:
                if steps:
                    results.append(Path(steps=list(steps)))
                return
            for rel, other in outs:
                if other in visited or depth >= hops:
                    if steps:
                        step = f"{name_of(current)} --{rel}--> {name_of(other)}"
                        results.append(Path(steps=steps + [step]))
                    continue
                step = f"{name_of(current)} --{rel}--> {name_of(other)}"
                walk(other, steps + [step], visited | {other}, depth + 1)

        walk(start, [], {start}, 0)
        return results

    def remove_entity(self, node_id: str) -> int:
        """Delete a node and every edge touching it. Returns edges removed."""
        cur = self._conn.execute(
            "DELETE FROM edges WHERE src = ? OR dst = ?", (node_id, node_id)
        )
        edges_removed = cur.rowcount
        self._conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        self._conn.commit()
        return edges_removed

    # --- stats ---
    def node_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    def edge_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0])

    def relation_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT rel, COUNT(*) c FROM edges GROUP BY rel"
        ).fetchall()
        return {r["rel"]: int(r["c"]) for r in rows}
