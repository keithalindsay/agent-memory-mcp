"""FastMCP server exposing the seven memory tools (DESIGN.md §7.1).

Each tool is a thin adapter over a process-lifetime `Memory`, validating input
and returning a plain JSON-serializable dict (no pydantic objects cross the MCP
boundary). Tool docstrings are what the model reads — keep them concrete.
"""

from __future__ import annotations

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .config import load_config

mcp = FastMCP("agent-memory-mcp")

# Overrides set by cli.serve() before main() runs.
_overrides: dict[str, Optional[str]] = {"db": None, "extractor": None, "embedder": None}
_memory = None


def configure(
    db: Optional[str] = None, extractor: Optional[str] = None, embedder: Optional[str] = None
) -> None:
    """Set config overrides (called by the CLI before serving) and reset memory."""
    global _memory
    _overrides["db"] = db
    _overrides["extractor"] = extractor
    _overrides["embedder"] = embedder
    _memory = None


def get_memory():
    """Lazily build the process-lifetime Memory from config + overrides."""
    global _memory
    if _memory is not None:
        return _memory
    from .embedders import get_embedder
    from .extractors import get_extractor
    from .memory import Memory

    cfg = load_config()
    db = _overrides["db"] or cfg.db
    extractor_name = _overrides["extractor"] or cfg.extractor
    embedder_name = _overrides["embedder"] or cfg.embedder
    if extractor_name == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        # Never crash the server: fall back to the offline rules extractor.
        extractor_name = "rules"
    emb = get_embedder(embedder_name, dim=cfg.dim)
    ext = get_extractor(extractor_name, model=cfg.model)
    _memory = Memory(db=db, embedder=emb, extractor=ext, dim=cfg.dim, k=cfg.k)
    return _memory


@mcp.tool()
def remember(text: str) -> dict:
    """Save something to long-term memory. Pass a plain-language statement or a
    short paragraph — e.g. 'Dana manages Alex, who owns the billing service.'
    You do NOT need to format it; just say what's true."""
    return get_memory().remember(text).model_dump()


@mcp.tool()
def recall(query: str, method: str = "auto") -> dict:
    """Answer a question from long-term memory. Ask in plain language — e.g.
    'Who does Dana's manager report to?'. method='auto' (default) tries the
    knowledge graph first and falls back to vector similarity; method='graph'
    or method='vector' pins a single backend (used for head-to-head
    benchmarking)."""
    return get_memory().recall(query, method=method).model_dump()


@mcp.tool()
def compare(query: str) -> dict:
    """THE benchmarking tool. Answer the SAME question with BOTH backends and
    show them side by side, so you can see where the knowledge graph beats plain
    vector RAG. Ask in plain language."""
    return get_memory().compare(query)


@mcp.tool()
def neighbors(entity: str) -> dict:
    """List what is directly connected to an entity (its immediate
    relationships)."""
    return get_memory().neighbors(entity)


@mcp.tool()
def entities(filter: str = "") -> dict:
    """List known entities, optionally filtered by a name substring."""
    return get_memory().entities(filter)


@mcp.tool()
def forget(subject: str) -> dict:
    """Remove an entity and all facts mentioning it from memory."""
    return get_memory().forget(subject)


@mcp.tool()
def stats() -> dict:
    """Summarize the memory: entity/edge/fact counts and relation types in
    use."""
    return get_memory().stats()


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
