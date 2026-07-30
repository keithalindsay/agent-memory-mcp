# agent-memory-mcp

> **A benchmarking MCP server for agent memory** — the same facts stored as both
> vector retrieval and a knowledge graph, behind identical tools, so you can measure
> where each one actually works instead of assuming.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Offline by default](https://img.shields.io/badge/runs-offline%2C%20no%20key-brightgreen.svg)

## The experiment

"Knowledge graphs are the memory layer for agents" is an oft-repeated claim that teams
rarely get to *test* against their own agent's real usage. `agent-memory-mcp` turns it
into a running experiment: it exposes two memory backends — **vector retrieval** and a
**knowledge graph** — behind the *same* Model Context Protocol tools, so the identical
question can be answered either way and the two compared side by side.
**Memory you can measure, not just trust.**

Single-shot vector retrieval works when the answer sits in one chunk. It struggles
whenever the answer must be *assembled by following relationships* — *"Who does Dana's
manager report to?"*, *"Which project uses a service owned by the Search Team?"* Those
answers live in no single chunk; a graph reaches them by traversal. This tool lets you
see where that happens, on your own data.

## ⚠️ What the benchmark measures — and what it doesn't

**Read this before quoting any number below.** The bundled harness is a *demonstration
on 17 questions*, not a benchmark, and the two arms are not matched:

- **The vector arm has no reader.** It returns the raw text of the top-1 retrieved
  chunk as its "answer" (`recall.py`), and the grader does loose containment. So its
  score is **retrieval precision@1, not question-answering accuracy.** A real RAG
  system is retrieval *plus* generation; the generation stage here (`phrase()`) is a
  documented no-op.
- **The graph arm gets a full query-planning layer** — entity resolution, relation
  inference from a keyword table, answer-type filtering. The vector arm gets
  `embed → top-1 → return the string`. The honest description of this comparison is
  **"a rule-based query planner over a knowledge graph, versus raw nearest-neighbour
  lookup"** — not "graph versus RAG."
- **Duplicate rows shrink the vector arm's budget.** Every statement is stored twice
  (templated fact + raw text), so 17 corpus lines become **27 rows**. At the default
  `k=4` the effective distinct-fact budget is about two.

### The numbers, with the caveat attached

Default `hash` embedder, 17 questions:

```
                Recall accuracy by category
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ method ┃ single_hop ┃ multi_hop ┃ aggregation ┃ overall ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ vector │       0.83 │      0.00 │        0.20 │    0.35 │
│ graph  │       1.00 │      1.00 │        1.00 │    1.00 │
└────────┴────────────┴───────────┴─────────────┴─────────┘
```

**That 0.35 is not the interesting number.** Score the same retriever on *recall@k* —
did the gold answer appear anywhere in the top k, which is the ceiling any reader could
reach — and the gap mostly closes:

| vector arm, same embedder and corpus | single_hop | multi_hop | aggregation | overall |
|---|---|---|---|---|
| as shipped (top-1 chunk as the answer) | 0.83 | 0.00 | 0.20 | **0.35** |
| recall@4 *(the default `k`)* | 1.00 | 0.00 | 0.40 | **0.47** |
| recall@8 | 1.00 | 0.67 | 0.60 | **0.76** |
| recall@17 | 1.00 | 1.00 | 0.80 | **0.94** |
| recall@27 *(the whole store)* | 1.00 | 1.00 | 1.00 | **1.00** |

**So the apparent 0.65-point gap is largely a missing reader and too small a `k`.**

### With real embeddings

The default embedder is a dependency-free **bag-of-hashed-tokens** — SHA-256 each token
into one of 256 buckets, count, L2-normalize. It has **no semantic capability**:
"manager" and "managed" are orthogonal. Re-run with `--embedder st`
(`all-MiniLM-L6-v2`):

| embedder | single_hop | multi_hop | aggregation |
|---|---|---|---|
| `hash` | 0.83 | 0.00 | 0.20 |
| `st` (all-MiniLM-L6-v2) | **1.00** | 0.00 | 0.00 |

A real embedding model takes single-hop to perfect and leaves multi-hop at zero. **That
is the finding worth keeping:** the multi-hop shortfall is structural to single-shot
retrieval, not an artifact of a weak embedder — while the headline "overall" gap is
mostly an artifact of the harness.

Run it yourself:

```bash
agent-memory-mcp bench                 # default hash embedder
agent-memory-mcp bench --embedder st   # requires: pip install -e ".[st]"
```

## Known limitations

Honest inventory, current as of 2026-07-30:

- **Traversal is direction-blind.** Edges are explored in both directions, so
  `MANAGED_BY` is effectively undirected: *"who is X's manager"* and *"who does X
  manage"* return the same node. The backend never answers "unknown" — it returns the
  inverse, and prints a correct-looking path underneath it. **This is a correctness
  bug, not a tuning issue.**
- **Hop depth is counted from the question text**, not discovered by search — the
  number of relation keywords in the query sets the BFS depth.
- **The rules extractor is a fixed trigger table** (~30 surface phrases over 7
  relations, in `vocab.py`). Unlisted phrasings silently produce no edge — *"Dianne
  leads the Discovery Team"* stores raw text and extracts nothing. The tool reports
  this in its `note`, but graph coverage is bounded by that table. On held-out
  paraphrases of the seed questions, graph accuracy drops well below the 1.00 above.
  It also ignores negation and drops conjunctions.
- **`compare()["agree"]` is almost always `false`** — it compares the graph's entity
  name against the vector arm's whole sentence, so the two rarely compare equal even
  when both are right.
- **`bench --db` is accepted and ignored** — the eval always runs in `:memory:`.
- **n = 17.** Fictional org-chart data, one domain, one sentence per chunk. This says
  nothing about chunking, which is where most real retrieval failures live.

## `compare` in 30 seconds

`compare` answers the **same** question with **both** backends, side by side:

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
  "vector": { "answer": "Dana is managed by Evan.",
              "support": ["Dana is managed by Evan.", "Evan is managed by Farah."] },
  "agree": false,
  "note": "Graph traversed 2 hops to reach the answer; vector similarity stopped at the single closest chunk."
}
```

The graph reaches **Farah** by following two `MANAGED_BY` edges and shows the path.
Note what the vector arm actually did: it **retrieved both hops** into `support` — the
information was there — and then returned only the top-1 chunk as its answer, because
there is no reader to assemble them. That is the missing-reader problem above, visible
in a single call.

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
- **Vector store** — `facts` + float32 vectors, brute-force cosine over L2-normalized
  embeddings. The raw statement is *always* embedded too (even when no relation could
  be extracted), so nothing is ever lost: graph when it can, vector when it can't.

`recall(method=...)` pins a single backend (`graph` | `vector`) for head-to-head
comparison, or `auto` (default) tries the graph first and falls back to vectors.
Full design: [DESIGN.md](DESIGN.md).

## Going further

- **Real embeddings** — install the `st` extra for sentence-transformers and pass
  `--embedder st`: `pip install -e ".[st]"`. Optional; the default hash embedder needs
  no download. Note: don't mix embedders against one persistent db — the stored vectors
  are fixed-dimension and the dimensions differ.
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
