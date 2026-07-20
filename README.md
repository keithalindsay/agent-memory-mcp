# agent-memory-mcp

> **A benchmarking MCP server for agent memory** — the same facts stored as both
> vector RAG and a knowledge graph, so you can measure, live, exactly where the
> graph beats vectors.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Offline by default](https://img.shields.io/badge/runs-offline%2C%20no%20key-brightgreen.svg)

## The experiment

"Knowledge graphs are the memory layer for agents" is an oft-repeated claim that
teams rarely get to *test* against their own agent's real usage. `agent-memory-mcp`
turns it into a running experiment: it exposes two memory backends — plain **vector
RAG** and a **knowledge graph** — behind the *same* Model Context Protocol tools, so
the identical question can be answered by vector similarity or by graph traversal and
the two compared head-to-head. **Memory you can benchmark, not just trust.**

Vector search works when the answer sits in one chunk. It fails whenever the answer
must be *assembled by following relationships* — *"Who does Dana's manager report
to?"*, *"Which project uses a service owned by the Search Team?"* Those answers live
in no single chunk; a graph answers them by traversal. This tool lets you see exactly
where that happens, on your own data.

## Screenshot

The bundled `bench` harness ingests a labeled seed set through the real `remember`
pipeline and scores graph-only vs vector-only recall by question category:

![bench scorecard](docs/screenshot-placeholder.png)

```
                Recall accuracy by category
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ method ┃ single_hop ┃ multi_hop ┃ aggregation ┃ overall ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ vector │       0.83 │      0.00 │        0.20 │    0.35 │
│ graph  │       1.00 │      1.00 │        1.00 │    1.00 │
└────────┴────────────┴───────────┴─────────────┴─────────┘
Wrote results/scorecard.json
```

Single-hop lookups are roughly a tie — both backends find a fact that sits in one
sentence. The graph pulls decisively ahead on **multi-hop** and **aggregation**
questions, exactly the relationship-following cases vector RAG can't assemble. Run it
yourself:

```bash
agent-memory-mcp bench
```

## `compare` in 30 seconds

`compare` is the headline tool: it answers the **same** question with **both**
backends, side by side. On a multi-hop question the graph traverses to the answer
while vector similarity stops at the closest single chunk:

```bash
agent-memory-mcp remember "Dana is managed by Evan. Evan is managed by Farah."
agent-memory-mcp compare "Who does Dana's manager report to?"
```

```json
{
  "query": "Who does Dana's manager report to?",
  "graph":  { "answer": "Farah",
              "support": ["Dana is managed by Evan.", "Evan is managed by Farah."],
              "path": ["Dana --MANAGED_BY--> Evan", "Evan --MANAGED_BY--> Farah"] },
  "vector": { "answer": "Dana is managed by Evan.", "support": ["Dana is managed by Evan."] },
  "agree": false,
  "note": "Graph traversed 2 hops to reach the answer; vector similarity stopped at the single closest chunk."
}
```

The graph reaches **Farah** by following two `MANAGED_BY` edges; vector RAG returns
the single closest sentence and never assembles the two hops.

## Add to your MCP client

The server speaks stdio and runs **fully offline with no API key** on its defaults
(deterministic rules extractor + hash embedder). Wire it in once and use it as live
memory *and* call `compare` mid-conversation.

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory-mcp",
      "args": ["serve"]
    }
  }
}
```

**Claude Code** — one command:

```bash
claude mcp add agent-memory -- agent-memory-mcp serve
```

The seven tools the model sees: `remember`, `recall(query, method="auto")`,
`compare`, `neighbors`, `entities`, `forget`, `stats`. Every tool takes **plain
language** — the agent never constructs a triple. `remember` never throws on
unparseable prose (it stores the raw text for vector fallback and says so);
`recall` returns `answer: null, method: "none"` on a miss rather than erroring.

## Quickstart (CLI, no keys)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e .

# teach it something — plain language, nothing to format
agent-memory-mcp remember "Dana is on the Search Team, which owns the Index Service. Atlas uses the Index Service."

# ask in plain language; method=auto tries the graph first, falls back to vectors
agent-memory-mcp recall "What does the Search Team own?"
agent-memory-mcp recall "Who does Dana's manager report to?" --method graph --show-path

# head-to-head on any question
agent-memory-mcp compare "Which project uses a service owned by the Search Team?"

# inspect and benchmark
agent-memory-mcp entities --filter service
agent-memory-mcp stats
agent-memory-mcp bench
```

Defaults: `--db ~/.agent-memory-mcp/memory.sqlite` (memory survives restarts;
`:memory:` for ephemeral), `--extractor rules`, `--embedder hash`, `--k 4`.

## How it works

Every fact is stored **both ways** in one SQLite file:

- **Knowledge graph** — `nodes` + `edges`. `remember` splits text into statements, an
  **extractor** turns each into entities and typed relations (`MANAGED_BY`, `OWNS`,
  `USES`, `MEMBER_OF`, `IS_A`, `LOCATED_IN`, `RELATED_TO`), and those are upserted as
  nodes and edges. `recall` resolves the entities in your question, infers the
  relation(s), and traverses for an answer **plus the supporting path** — so answers
  are explainable.
- **Vector store** — `facts` + float32 vectors. The raw statement is *always* embedded
  too (even when no relation could be extracted), so nothing is ever lost: graph when
  it can, vector when it can't.

`recall(method=...)` pins a single backend (`graph` | `vector`) for head-to-head
benchmarking, or `auto` (default) tries the graph first and falls back to vectors.
Full design: [DESIGN.md](DESIGN.md).

## Going further

- **Real embeddings** — install the `st` extra for sentence-transformers and pass
  `--embedder st`: `pip install -e ".[st]"`. Optional; the default hash embedder needs
  no download.
- **LLM extraction** — install the `llm` extra (`pip install -e ".[llm]"`) and run with
  `--extractor llm` to have Anthropic Claude turn messy prose into clean
  entities/relations. Requires `ANTHROPIC_API_KEY`; the model id is read only from
  config/env (`AGENT_MEMORY_MODEL`, default `claude-haiku-4-5`). The default `rules`
  extractor keeps the server keyless and deterministic.
- **Extend the ontology** — add a row to the relation vocabulary in
  `agent_memory_mcp/vocab.py` (trigger phrases + a fact template) to teach the rules
  extractor a new relation.

## Clean-room note

This is an original, clean-room, **generic** MCP server authored by Keith Lindsay. It
does **not** reference, reproduce, or depend on any specific employer's (including
Aerospike's) source code, proprietary data, internal metrics, benchmarks, product
names, schemas, or confidential specifics, and it is **not coupled to any particular
graph or vector database product**. The only thing proprietary about the author's
prior work was its integration with a specific product; the knowledge-graph technique
itself is general, publicly-known computer science. All datasets here are synthetic
and fictional.

## License

MIT © 2026 Keith Lindsay. See [LICENSE](LICENSE).
