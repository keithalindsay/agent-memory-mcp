"""Text embedders (DESIGN.md §6).

Default is a deterministic, dependency-free hashing embedder so the server runs
fully offline with no model download. An optional sentence-transformers embedder
is available via the ``st`` extra and is import-guarded.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common English function words carry no topical signal; dropping them keeps the
# bag-of-hashed-tokens cosine driven by content words, so an off-topic query does
# not score highly just by sharing "is/the/of".
_STOPWORDS = frozenset("""
    a an the and or but of to in on at by for from with as is are was were be been being
    do does did done have has had this that these those it its it's i you we he she they them
    me my our your their his her who whom whose what which where when why how
    about into over under out up down off no not so than then too very can could would should
    will shall may might must know knows tell tells about us
    """.split())


def _tokenize(text: str) -> list[str]:
    toks = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
    return toks or _TOKEN_RE.findall(text.lower())


@runtime_checkable
class Embedder(Protocol):
    """Protocol every embedder implements."""

    dim: int

    def embed(self, text: str) -> np.ndarray:
        """Return an L2-normalized float32 vector of length ``dim``."""
        ...


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedder.

    Each token is hashed into one of ``dim`` buckets; bucket counts form the
    vector, which is then L2-normalized. Identical text always yields an
    identical vector, so all non-LLM tests are reproducible.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokenize(text):
            h = hashlib.sha256(tok.encode("utf-8")).digest()
            bucket = int.from_bytes(h[:8], "big") % self.dim
            vec[bucket] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


class SentenceTransformerEmbedder:
    """Optional real embeddings via sentence-transformers (``st`` extra)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dim: int | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "The 'st' embedder requires sentence-transformers. "
                "Install it with: pip install 'agent-memory-mcp[st]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dim = dim or self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:  # pragma: no cover - optional dependency
        vec = np.asarray(self._model.encode(text), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec


def get_embedder(name: str = "hash", dim: int = 256) -> Embedder:
    """Factory: return an embedder by name (``hash`` default, ``st`` optional)."""
    name = (name or "hash").lower()
    if name == "hash":
        return HashEmbedder(dim=dim)
    if name == "st":
        return SentenceTransformerEmbedder(dim=dim)
    raise ValueError(f"Unknown embedder: {name!r} (expected 'hash' or 'st')")
