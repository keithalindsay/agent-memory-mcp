"""Typer CLI for agent-memory-mcp (scaffold stub — filled in at build step 10)."""

import typer

app = typer.Typer(
    name="agent-memory-mcp",
    help="A benchmarking MCP server for agent memory (vector RAG vs knowledge graph).",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the version."""
    from agent_memory_mcp import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
