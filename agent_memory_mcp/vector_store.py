"""Brute-force cosine vector store over SQLite (DESIGN.md §8.2)."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Optional

import numpy as np

from .models import Hit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
  fact_id TEXT PRIMARY KEY,
  text    TEXT NOT NULL,
  src     TEXT, rel TEXT, dst TEXT,
  vector  BLOB NOT NULL
);
"""


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def fact_id_for(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


class VectorStore:
    """Facts + float32 vectors in SQLite; cosine top-k via numpy."""

    def __init__(
        self, db_path: str, dim: int = 256, conn: Optional[sqlite3.Connection] = None
    ) -> None:
        self.dim = dim
        self._conn = conn or sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def add(
        self,
        fact_id: Optional[str],
        text: str,
        src: Optional[str],
        rel: Optional[str],
        dst: Optional[str],
        vector: np.ndarray,
    ) -> str:
        """Insert (or upsert) a fact and its vector. Returns the fact_id."""
        fid = fact_id or fact_id_for(text)
        blob = np.asarray(vector, dtype=np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO facts (fact_id, text, src, rel, dst, vector) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(fact_id) DO UPDATE SET text=excluded.text, src=excluded.src, "
            "rel=excluded.rel, dst=excluded.dst, vector=excluded.vector",
            (fid, text, src, rel, dst, blob),
        )
        self._conn.commit()
        return fid

    def search(self, query_vec: np.ndarray, k: int = 4) -> list[Hit]:
        """Return the top-k facts by cosine similarity (vectors are L2-normalized)."""
        rows = self._conn.execute("SELECT fact_id, text, vector FROM facts").fetchall()
        if not rows:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = float(np.linalg.norm(q))
        if qn > 0.0:
            q = q / qn
        mat = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
        scores = mat @ q
        order = np.argsort(-scores)[:k]
        return [
            Hit(fact_id=rows[i]["fact_id"], text=rows[i]["text"], score=float(scores[i]))
            for i in order
        ]

    def delete_by_entity(self, entity_id: str, name: Optional[str] = None) -> int:
        """Delete facts whose src/dst equals the entity id, or (optionally) whose
        raw text mentions ``name``. Returns count removed."""
        if name:
            cur = self._conn.execute(
                "DELETE FROM facts WHERE src = ? OR dst = ? OR text LIKE ?",
                (entity_id, entity_id, f"%{name}%"),
            )
        else:
            cur = self._conn.execute(
                "DELETE FROM facts WHERE src = ? OR dst = ?", (entity_id, entity_id)
            )
        self._conn.commit()
        return cur.rowcount

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])

    def relation_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT rel, COUNT(*) c FROM facts WHERE rel IS NOT NULL GROUP BY rel"
        ).fetchall()
        return {r["rel"]: int(r["c"]) for r in rows}
