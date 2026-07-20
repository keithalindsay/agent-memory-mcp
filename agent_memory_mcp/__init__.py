"""agent-memory-mcp: a benchmarking MCP server for agent memory.

The same facts are stored both as vector RAG and as a knowledge graph, so you can
measure — live, on your own data — exactly where the graph beats vectors.
"""

from .memory import Memory

__version__ = "0.1.0"

__all__ = ["Memory", "__version__"]
