"""Typer CLI for agent-memory-mcp (DESIGN.md §7.2).

`serve` is what an MCP client launches. The other subcommands let a human seed,
inspect, and benchmark memory without an MCP client.
"""

from __future__ import annotations

import json as _json
import os
from typing import Optional

import typer

from .config import load_config

app = typer.Typer(
    name="agent-memory-mcp",
    help="A benchmarking MCP server for agent memory (vector RAG vs knowledge graph).",
    no_args_is_help=True,
    add_completion=False,
)


def _build_memory(db: Optional[str], extractor: Optional[str], embedder: Optional[str]):
    from .embedders import get_embedder
    from .extractors import get_extractor
    from .memory import Memory

    cfg = load_config()
    db = db or cfg.db
    extractor_name = extractor or cfg.extractor
    embedder_name = embedder or cfg.embedder

    if extractor_name == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        typer.secho(
            "error: --extractor llm requires ANTHROPIC_API_KEY to be set.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)

    emb = get_embedder(embedder_name, dim=cfg.dim)
    ext = get_extractor(extractor_name, model=cfg.model)
    return Memory(db=db, embedder=emb, extractor=ext, dim=cfg.dim, k=cfg.k)


def _echo_json(obj) -> None:
    typer.echo(_json.dumps(obj, indent=2, ensure_ascii=False))


@app.command()
def serve(
    db: Optional[str] = typer.Option(None, help="Path to the SQLite memory file."),
    extractor: Optional[str] = typer.Option(None, help="rules | llm"),
    embedder: Optional[str] = typer.Option(None, help="hash | st"),
) -> None:
    """Start the MCP server over stdio (what an MCP client launches)."""
    if extractor == "llm" and not os.environ.get("ANTHROPIC_API_KEY"):
        typer.secho(
            "error: --extractor llm requires ANTHROPIC_API_KEY to be set.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=2)
    from . import server

    server.configure(db=db, extractor=extractor, embedder=embedder)
    server.main()


@app.command()
def remember(
    text: str = typer.Argument(..., help="A plain-language statement or short paragraph."),
    db: Optional[str] = typer.Option(None),
    extractor: Optional[str] = typer.Option(None, help="rules | llm"),
) -> None:
    """Save something to memory (extract + embed)."""
    mem = _build_memory(db, extractor, None)
    result = mem.remember(text)
    _echo_json(result.model_dump())


@app.command()
def recall(
    query: str = typer.Argument(..., help="A plain-language question."),
    db: Optional[str] = typer.Option(None),
    k: int = typer.Option(4, help="Top-k for vector fallback."),
    method: str = typer.Option("auto", help="auto | graph | vector"),
    show_path: bool = typer.Option(False, "--show-path", help="Print the graph path."),
) -> None:
    """Answer a question from memory."""
    mem = _build_memory(db, None, None)
    result = mem.recall(query, method=method, k=k)
    out = result.model_dump()
    if not show_path:
        out.pop("path", None)
    _echo_json(out)


@app.command()
def compare(
    query: str = typer.Argument(..., help="A plain-language question."),
    db: Optional[str] = typer.Option(None),
    k: int = typer.Option(4),
) -> None:
    """Answer the SAME question with both backends, side by side."""
    mem = _build_memory(db, None, None)
    _echo_json(mem.compare(query, k=k))


@app.command()
def entities(
    filter: str = typer.Option("", "--filter", help="Name substring filter."),
    db: Optional[str] = typer.Option(None),
) -> None:
    """List known entities."""
    mem = _build_memory(db, None, None)
    _echo_json(mem.entities(filter))


@app.command()
def forget(
    subject: str = typer.Argument(..., help="Entity to remove."),
    db: Optional[str] = typer.Option(None),
) -> None:
    """Remove an entity and all facts mentioning it."""
    mem = _build_memory(db, None, None)
    _echo_json(mem.forget(subject))


@app.command()
def stats(db: Optional[str] = typer.Option(None)) -> None:
    """Summarize memory: entity/edge/fact counts and relation types."""
    mem = _build_memory(db, None, None)
    _echo_json(mem.stats())


@app.command()
def bench(
    corpus: Optional[str] = typer.Option(None, help="Path to seed corpus (.txt)."),
    questions: Optional[str] = typer.Option(None, help="Path to labeled questions (.json)."),
    db: str = typer.Option(":memory:", help="DB to ingest into (ephemeral by default)."),
    embedder: Optional[str] = typer.Option(None, help="hash | st"),
    out: str = typer.Option("results/scorecard.json", help="Where to write the scorecard JSON."),
) -> None:
    """Ingest the seed set, run graph-vs-vector recall, print a scorecard."""
    from . import bench as bench_mod

    bench_mod.run_bench(
        corpus_path=corpus,
        questions_path=questions,
        db=db,
        embedder=embedder,
        out_path=out,
    )


if __name__ == "__main__":
    app()
