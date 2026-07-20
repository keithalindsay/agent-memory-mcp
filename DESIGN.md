# DESIGN.md — agent-memory-mcp

> Build-ready design package. An expert coding agent should be able to build the v1 MVP from this
> document alone. Read top-to-bottom before writing code.
>
> Author: Keith Lindsay (GitHub: [keithalindsay](https://github.com/keithalindsay)) · License: MIT © 2026 Keith Lindsay

---

## 1. What it is (one-liner) + the problem it solves

**agent-memory-mcp** is a **benchmarking MCP server for agent memory**: it gives an AI agent two memory
backends — plain **vector RAG** and a **knowledge graph** — behind the *same* Model Context Protocol tools,
and lets you **measure, live and on your own data, where the graph beats vectors**. The agent remembers and
recalls in plain language; under the hood every fact is stored *both* ways, so the same question can be
answered by vector similarity or by graph traversal and the two compared head-to-head.

**The experiment it runs.** An agent's "memory" is usually a vector store: embed text chunks, retrieve by
similarity. That works when the answer sits in one chunk, but fails whenever the answer must be *assembled
by following relationships*: *"Who does Dana's manager report to?"*, *"Which services depend on the one
Alex owns?"* Those answers live in no single chunk — a graph answers them by traversal. This is an oft-
repeated claim ("knowledge graphs are the memory layer") that teams rarely get to *test* against their own
agent's real usage. `agent-memory-mcp` turns it into a running experiment: wire it into any MCP client, let
the agent use it as memory, and use the built-in `bench`/`compare` tooling to see exactly where graph
recall wins, ties, or loses versus vector RAG. **Memory you can benchmark, not just trust.**

---

## 2. Who it's for and why it's useful

- **AI engineers / PMs / researchers** who want to *test* the "knowledge graphs vs vector RAG" claim
  against a real agent's usage — a running, measurable experiment instead of a slide. This is the primary
  audience: the tool is a benchmark harness first.
- **Agent builders** (Claude Desktop, Claude Code, Cursor, or any MCP client) who, as a by-product, get
  durable memory that survives sessions and can answer multi-hop, relationship questions.
- **Tech writers / speakers** who need a clean, MIT-licensed reference they can point at and reproduce.

Why it's useful:

- **Nothing to format.** The agent never emits a rigid triple. `remember` and `recall` each take a single
  natural-language string; the server does entity/relation extraction and traversal internally. This is a
  deliberate robustness choice — LLM-generated structured arguments drift and malform over time; plain
  prose does not.
- **Never loses a fact.** If extraction can't confidently structure a statement, the raw sentence is still
  embedded and stored, so `recall` can find it via vector search. Graph when it can, vector when it can't.
- **Explainable.** `recall` returns the supporting subgraph / path, so the agent (and you) can see *why* an
  answer was produced.
- **Offline by default, zero keys.** The default extractor and embedder are deterministic and local — the
  server runs with no API key and no model download. LLM-quality extraction is an opt-in upgrade.
- **Measurable.** A bundled eval harness scores graph-vs-vector recall on a labeled question set, printing
  a scorecard — the honest "here's where the graph actually wins" artifact.

---

## 3. Scope: v1 MVP vs out-of-scope

### v1 MVP (must-have)

1. **MCP server over stdio** exposing the tools in §7.1 (`remember`, `recall`, `compare`, `neighbors`,
   `entities`, `forget`, `stats`). Built on the official Python MCP SDK (`FastMCP`). `compare` (answer the
   same query with both backends, side by side) is the headline benchmarking tool.
2. **Free-text ingestion.** `remember(text)` splits input into statements and runs an **extractor** to
   produce entities + typed relations, writing them into the graph store; the raw statement is always also
   embedded into the vector store.
3. **Two extractors behind one interface:**
   - `rules` (default) — deterministic, offline, dependency-free. Pattern/verb-based extraction over a
     known relation vocabulary (§8.4). Honest about its limits; unmatched text still gets stored for vector
     recall.
   - `llm` (optional) — Anthropic Claude turns messy prose into clean entities/relations. Needs a key;
     non-deterministic. Enabled by config/env.
4. **Hybrid recall.** `recall(query)` resolves entities in the question, infers the relation(s), traverses
   the graph for an answer + path; if the graph yields nothing, falls back to top-k vector search. Returns
   answer, method used (`graph` | `vector` | `none`), supporting facts, and the path when graph-answered.
5. **Two SQLite-backed stores** in one database file: a brute-force cosine **vector store** and a
   `nodes`+`edges` **graph store** (schema in §8.2). Reused design from `agent-memory-starter`.
6. **Deterministic local embedder** (`HashEmbedder`, dim=256) as default; optional
   `sentence-transformers` embedder as an extra.
7. **Persistence.** Memory persists to a db file (default `~/.agent-memory-mcp/memory.sqlite`, overridable)
   so an agent's memory survives restarts. `:memory:` supported for tests/ephemeral use.
8. **A thin CLI** (`agent-memory-mcp`) with `serve` (start the MCP server), plus `remember`, `recall`,
   `entities`, `forget`, and `bench` subcommands for local testing/inspection *without* an MCP client.
9. **Eval harness** (`bench`) — ingest a labeled seed set, run graph vs vector recall, print a Rich
   scorecard + write `results/scorecard.json`.
10. **Config**: env + optional `agent-memory-mcp.toml`. Model id read from env, never hardcoded.
11. **README** with a copy-paste MCP client config (Claude Desktop / Claude Code) and a `## Screenshot`
    placeholder; **tests** (pytest) covering extraction, stores, traversal, recall, and the tools.

### Explicitly OUT of scope for v1

- Any non-SQLite backend (Neo4j, pgvector, Pinecone, **or any vendor-specific graph/vector database**). v1
  is SQLite + numpy only. *(The tool is deliberately backend-agnostic and coupled to no specific product.)*
- Multi-user / auth / a hosted service / network transport other than the selected LLM provider. stdio only.
- Agent planning / tool-use loops. The server answers single calls.
- Automatic ontology learning, entity resolution across synonyms beyond simple normalized name matching,
  coreference resolution, temporal/versioned facts, contradiction resolution.
- Embedding training / fine-tuning / re-ranking. Streaming. Non-text modalities.

---

## 4. Tech stack + rationale

| Concern | Choice | Rationale |
|---|---|---|
| Language | **Python 3.10+** | Expected for an MCP/memory/RAG tool; official MCP SDK is first-class in Python. |
| MCP layer | **`mcp` SDK (`FastMCP`)** | Official Model Context Protocol server SDK; decorator-based tool registration; stdio transport that Claude Desktop/Code speak natively. |
| Storage | **SQLite (stdlib `sqlite3`)** | Zero-install, file-based, embeddable; one db file holds both vector and graph tables. Honest local/persistent story. |
| Vector math | **numpy** | Cosine similarity without a vector DB. |
| CLI | **Typer** | Typed CLI + generated help; matches sibling repos' house style. |
| Tables | **Rich** | Pretty scorecard for the eval screenshot. |
| Models | **pydantic v2** | Validated data models for tool inputs/outputs, corpus, results. |
| Tests | **pytest** | Standard. |
| Embeddings (opt) | **sentence-transformers** (extra) | Real embeddings when wanted; not required to run. |
| LLM (opt) | **anthropic** (extra) | Optional `llm` extractor + optional LLM answer phrasing; deterministic `rules` default so the server runs keyless. |

**Defaults are dependency-light and deterministic** (hash embedder + rules extractor) so
`pip install -e .` → `agent-memory-mcp serve` works offline immediately. Quality upgrades are opt-in extras.

---

## 5. Architecture overview

```
   MCP client (Claude Desktop / Code / Cursor)
                    │  stdio (MCP protocol)
                    ▼
        ┌──────────────────────────────┐
        │   server.py  (FastMCP app)    │  tools: remember | recall | compare |
        │   thin adapters over Memory   │  neighbors | entities | forget | stats
        └───────────────┬──────────────┘
                        │
                 ┌──────▼───────┐
                 │  memory.py    │  the façade: remember(text) / recall(query) /
                 │  (Memory)     │  neighbors / entities / forget / stats
                 └──┬────────┬───┘
          ┌─────────┘        └──────────┐
          ▼                             ▼
   Extractor (rules|llm)          Recall pipeline
   text → [Statement→edges]       entity match → relation infer →
          │                       graph traverse → (fallback) vector search
          ▼                             │
   ┌─────────────┐  ┌──────────────┐    │
   │ GraphStore  │  │ VectorStore  │◄───┘
   │ nodes+edges │  │ facts+vecs   │
   │  (SQLite)   │  │  (SQLite)    │   Embedder (hash|st)
   └─────────────┘  └──────────────┘
          │                │
          └──────── one memory.sqlite ────────┘

   CLI (Typer): serve | remember | recall | entities | forget | bench
   Eval: bench.py  → Rich scorecard + results/scorecard.json
```

**Flow — `remember(text)`:** split into statements → for each, extractor yields `(entities, relations)` →
upsert nodes/edges into GraphStore → embed the raw statement and add to VectorStore (always, even when no
relation was extracted). Return a summary of what was learned.

**Flow — `recall(query)`:** find candidate entities in the query via `GraphStore.find_nodes` → infer
relation(s) from query keywords → `traverse` for an answer + path → if empty, embed query and `VectorStore.search`
top-k → phrase an answer (rules by default; optional LLM phrasing) → return `{answer, method, support, path}`.

---

## 6. Key components / modules

- `server.py` — the `FastMCP` app. Each MCP tool is a thin, well-documented adapter that validates input,
  calls `Memory`, and returns a JSON-serializable dict. Holds the tool docstrings the *model* reads (these
  matter — see §7.1). `main()` starts stdio transport.
- `memory.py` — `Memory` façade tying stores + extractor + recall together. The single object the server
  and CLI both use. Methods mirror the tools.
- `extractors.py` — `Extractor` protocol; `RulesExtractor` (default, deterministic pattern/verb matcher
  over the relation vocabulary in §8.4); `LlmExtractor` (Anthropic; structured extraction prompt; model id
  from env). Factory `get_extractor(name)`.
- `embedders.py` — `Embedder` protocol; `HashEmbedder` (deterministic, dim=256, L2-normalized bag-of-hashed-tokens);
  optional `SentenceTransformerEmbedder`. Factory `get_embedder(name, dim)`. *(Ported design from
  agent-memory-starter §6.)*
- `vector_store.py` — `VectorStore` over SQLite: `add(fact_id, text, src, rel, dst, vector)`,
  `search(query_vec, k) -> list[Hit]`, `delete_by_entity(name)`. Brute-force cosine via numpy.
- `graph_store.py` — `GraphStore` over SQLite: `add_node`, `add_edge`, `get_node`, `find_nodes(name)`
  (case-insensitive), `neighbors(node_id, rel=None, direction)`, `traverse(start, relation, hops)`,
  `remove_entity(name)`. Schema in §8.2.
- `recall.py` — the recall pipeline: `resolve_entities(query, graph)`, `infer_relation(query)`
  (keyword→rel map, §8.4), `graph_answer(...)`, `vector_answer(...)`, `phrase(answer, mode)`.
- `models.py` — pydantic models: `Entity`, `Relation`, `Statement`, `Fact`, `Hit`, `Path`,
  `RememberResult`, `RecallResult`, `Question`, `GradedResult`, `Scorecard` (§8.5).
- `config.py` — load env + optional `agent-memory-mcp.toml`; resolve db path, extractor, embedder, model env.
- `cli.py` — Typer app: `serve`, `remember`, `recall`, `entities`, `forget`, `bench`.
- `bench.py` — eval harness → Rich scorecard + `results/scorecard.json`.
- `seed/corpus.txt`, `seed/questions.json` — bundled natural-language facts + labeled questions for `bench`.

---

## 7. Interface: MCP tools / CLI / config

### 7.1 MCP tools (the primary interface)

Registered on `FastMCP`. **Every tool takes plain strings — the agent never constructs a triple.** The
tool descriptions below are the docstrings the model actually sees; keep them this concrete.

```
remember(text: str) -> dict
    "Save something to long-term memory. Pass a plain-language statement or a short
     paragraph — e.g. 'Dana manages Alex, who owns the billing service.' You do NOT
     need to format it; just say what's true."
    returns: {
      "learned": [ {"src": "...", "rel": "...", "dst": "..."} , ... ],   # relations added
      "entities_added": int,
      "relations_added": int,
      "stored_raw": int,          # statements embedded for vector fallback
      "note": str                 # human-readable summary, incl. anything not structured
    }

recall(query: str, method: str = "auto") -> dict
    "Answer a question from long-term memory. Ask in plain language — e.g.
     'Who does Dana's manager report to?'. method='auto' (default) tries the
     knowledge graph first and falls back to vector similarity; method='graph'
     or method='vector' pins a single backend (used for head-to-head benchmarking)."
    returns: {
      "answer": str | null,
      "method": "graph" | "vector" | "none",   # the backend that actually answered
      "support": [str, ...],      # supporting fact sentences
      "path": [str, ...]          # e.g. ["Dana --MANAGED_BY--> Evan", "Evan --MANAGED_BY--> Farah"]
    }

compare(query: str) -> dict
    "THE benchmarking tool. Answer the SAME question with BOTH backends and show
     them side by side, so you can see where the knowledge graph beats plain
     vector RAG. Ask in plain language."
    returns: {
      "query": str,
      "graph":  {"answer": str|null, "support": [str,...], "path": [str,...]},
      "vector": {"answer": str|null, "support": [str,...]},
      "agree": bool,              # do the two backends give the same normalized answer?
      "note": str                 # e.g. "Graph traversed 2 hops; vector stopped at the top chunk."
    }

neighbors(entity: str) -> dict
    "List what is directly connected to an entity (its immediate relationships)."
    returns: { "entity": str|null, "out": [{"rel","dst"}], "in": [{"rel","src"}] }

entities(filter: str = "") -> dict
    "List known entities, optionally filtered by a name substring."
    returns: { "entities": [{"id","type","name"}], "count": int }

forget(subject: str) -> dict
    "Remove an entity and all facts mentioning it from memory."
    returns: { "removed_entity": str|null, "edges_removed": int, "facts_removed": int }

stats() -> dict
    "Summarize the memory: entity/edge/fact counts and relation types in use."
    returns: { "entities": int, "edges": int, "facts": int, "relations": {rel: count} }
```

**Design rules for tools (enforced in code):**
- Be liberal in what you accept: `remember` never throws on unparseable prose — it stores the raw text and
  says so in `note`. `recall` never throws on a miss — it returns `answer: null, method: "none"`.
- Outputs are always JSON-serializable dicts (no pydantic objects leak across the MCP boundary).
- No secrets in any output.

### 7.2 CLI (entry point `agent-memory-mcp`, module `agent_memory_mcp.cli:app`)

```
agent-memory-mcp serve     [--db PATH] [--extractor rules|llm] [--embedder hash|st]
agent-memory-mcp remember  "TEXT"  [--db PATH] [--extractor rules|llm]
agent-memory-mcp recall    "QUERY" [--db PATH] [--k INT] [--method auto|graph|vector] [--show-path]
agent-memory-mcp compare   "QUERY" [--db PATH] [--k INT]   # graph vs vector, side by side
agent-memory-mcp entities  [--filter SUBSTR] [--db PATH]
agent-memory-mcp forget    "SUBJECT" [--db PATH]
agent-memory-mcp bench     [--corpus PATH] [--questions PATH] [--db PATH]
                           [--embedder hash|st] [--out results/scorecard.json]
```

Defaults: `--db ~/.agent-memory-mcp/memory.sqlite`, `--extractor rules`, `--embedder hash`, `--k 4`.
`serve` is what an MCP client launches (see README config). The other subcommands exist so a human can
seed/inspect memory and run the benchmark without an MCP client. `--extractor llm` requires
`ANTHROPIC_API_KEY`; missing ⇒ exit 2 with a clear message.

### 7.3 Config

Env first, optional `agent-memory-mcp.toml` in cwd for defaults:

```toml
[defaults]
db = "~/.agent-memory-mcp/memory.sqlite"
extractor = "rules"     # rules | llm
embedder  = "hash"      # hash | st
dim = 256
k = 4

[llm]
model_env = "AGENT_MEMORY_MODEL"   # env var holding the Anthropic model id (default: claude-haiku-4-5)
```

Env: `ANTHROPIC_API_KEY` (only for `llm` extractor/phrasing), `AGENT_MEMORY_MODEL` (model id),
`AGENT_MEMORY_DB` (db path override). CLI flags override the file; the file overrides built-in defaults.
**No model string is hardcoded outside `config.py`.**

---

## 8. Data models / schemas (concrete)

### 8.1 Relation vocabulary (the shared ontology)

A small, generic, closed set of relation types v1 understands, each with a surface-form trigger set (for
the rules extractor and the query relation-inference map) and a fact-sentence template:

| `rel` | Trigger phrases (extract & query) | Fact template `(src, dst)` |
|---|---|---|
| `MANAGED_BY` | "manages", "manager", "reports to", "managed by" | "{src} is managed by {dst}." |
| `MEMBER_OF` | "member of", "on the team", "part of", "belongs to" | "{src} is a member of {dst}." |
| `OWNS` | "owns", "owner of", "responsible for", "maintains" | "{src} owns {dst}." |
| `USES` | "uses", "depends on", "built on", "relies on" | "{src} uses {dst}." |
| `IS_A` | "is a", "is an", "role is", "works as" | "{src} is a {dst}." |
| `LOCATED_IN` | "located in", "based in", "in" (loc ctx) | "{src} is located in {dst}." |
| `RELATED_TO` | fallback for a detected S-V-O with unknown verb | "{src} is related to {dst}." |

Extending this table is the documented way to teach the rules extractor new relations. `RELATED_TO` keeps
the graph useful even for verbs outside the set (still better than losing the edge).

### 8.2 SQLite schema (one db file)

```sql
-- graph store
CREATE TABLE IF NOT EXISTS nodes (
  id    TEXT PRIMARY KEY,          -- "type:normalized-name", e.g. "person:dana"
  type  TEXT NOT NULL,             -- Person | Team | Service | Project | Thing (default)
  name  TEXT NOT NULL,             -- display name, e.g. "Dana"
  attrs TEXT NOT NULL DEFAULT '{}' -- JSON
);
CREATE TABLE IF NOT EXISTS edges (
  src TEXT NOT NULL,
  rel TEXT NOT NULL,
  dst TEXT NOT NULL,
  PRIMARY KEY (src, rel, dst),
  FOREIGN KEY (src) REFERENCES nodes(id),
  FOREIGN KEY (dst) REFERENCES nodes(id)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);

-- vector store
CREATE TABLE IF NOT EXISTS facts (
  fact_id TEXT PRIMARY KEY,        -- sha256 of normalized text
  text    TEXT NOT NULL,
  src     TEXT, rel TEXT, dst TEXT,-- nullable: raw statements with no extracted relation
  vector  BLOB NOT NULL            -- float32 numpy bytes, length = dim
);
```

Entity id normalization: `type:slug(name)` where `slug` lowercases and hyphenates. Unknown-type entities
default to type `Thing`. Re-remembering the same entity upserts (no duplicates); re-adding the same edge is
idempotent via the composite PK.

### 8.3 Seed corpus (`seed/corpus.txt`) — natural language, for `bench`

Plain sentences, one per line, e.g.:
```
Dana is an Engineer.
Dana is managed by Evan.
Evan is managed by Farah.
Dana is a member of the Search Team.
The Search Team owns the Index Service.
The Atlas project uses the Index Service.
```
`bench` ingests these through the **real** `remember` path (extractor included), so the benchmark measures
the actual pipeline, not a hand-built graph. All data is **synthetic and fictional**.

### 8.4 Questions file (`seed/questions.json`) — labeled eval set

```json
[
  { "id": "q1", "text": "What is Dana's role?",
    "category": "single_hop", "gold_answer": "Engineer" },
  { "id": "q2", "text": "Who does Dana's manager report to?",
    "category": "multi_hop", "gold_answer": "Farah" },
  { "id": "q3", "text": "Which project uses a service owned by the Search Team?",
    "category": "aggregation", "gold_answer": "Atlas" }
]
```
Categories: `single_hop` (vector ≈ graph), `multi_hop` / `aggregation` (graph should win). Ship ~12–18.

### 8.5 pydantic model shapes

```python
class Entity(BaseModel):    id: str; type: str; name: str; attrs: dict = {}
class Relation(BaseModel):  src: str; rel: str; dst: str
class Statement(BaseModel): text: str; entities: list[Entity] = []; relations: list[Relation] = []
class Fact(BaseModel):      fact_id: str; text: str; src: str|None; rel: str|None; dst: str|None
class Hit(BaseModel):       fact_id: str; text: str; score: float
class Path(BaseModel):      steps: list[str]              # ["Dana --MANAGED_BY--> Evan", ...]
class RememberResult(BaseModel): learned: list[Relation]; entities_added: int
                                 relations_added: int; stored_raw: int; note: str
class RecallResult(BaseModel):   answer: str|None; method: Literal["graph","vector","none"]
                                 support: list[str] = []; path: list[str] = []
class Question(BaseModel):  id: str; text: str
                            category: Literal["single_hop","multi_hop","aggregation"]; gold_answer: str
class GradedResult(BaseModel): question_id: str; method: str; category: str
                               predicted: str|None; gold: str; correct: bool
class Scorecard(BaseModel): by_method_category: dict[str, dict[str, float]]
                            overall: dict[str, float]; n: int; results: list[GradedResult]
```
MCP tool return dicts are `model_dump()` of the relevant models (plain JSON across the boundary).

---

## 9. End-to-end example

**Agent → `remember`:**
```
remember("Dana is an engineer. Dana is managed by Evan, and Evan is managed by Farah.
          Dana is on the Search Team, which owns the Index Service. Atlas uses the Index Service.")

→ { "learned": [
      {"src":"person:dana","rel":"IS_A","dst":"thing:engineer"},
      {"src":"person:dana","rel":"MANAGED_BY","dst":"person:evan"},
      {"src":"person:evan","rel":"MANAGED_BY","dst":"person:farah"},
      {"src":"person:dana","rel":"MEMBER_OF","dst":"team:search-team"},
      {"src":"team:search-team","rel":"OWNS","dst":"service:index-service"},
      {"src":"thing:atlas","rel":"USES","dst":"service:index-service"} ],
    "entities_added": 6, "relations_added": 6, "stored_raw": 4,
    "note": "Learned 6 relationships across 4 statements." }
```

**Agent → `recall` (multi-hop, graph wins):**
```
recall("Who does Dana's manager report to?")
→ { "answer": "Farah", "method": "graph",
    "support": ["Dana is managed by Evan.", "Evan is managed by Farah."],
    "path": ["Dana --MANAGED_BY--> Evan", "Evan --MANAGED_BY--> Farah"] }
```

**Agent → `recall` (no relation, vector fallback):**
```
recall("What do we know about the index service?")
→ { "answer": "The Search Team owns the Index Service.", "method": "vector",
    "support": ["The Search Team owns the Index Service.", "Atlas uses the Index Service."], "path": [] }
```

**Human → `bench` (the differentiator, illustrative):**
```
             Recall accuracy by category
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ method ┃ single_hop ┃ multi_hop┃ aggregation ┃ overall ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━┩
│ vector │       0.90 │     0.20 │        0.17 │    0.44 │
│ graph  │       0.90 │     1.00 │        0.83 │    0.91 │
└────────┴────────────┴──────────┴─────────────┴─────────┘
Wrote results/scorecard.json
```
(Exact numbers depend on the shipped seed set; single_hop ≈ tie, multi_hop/aggregation favor graph. Tests
assert this **directionally and loosely** — §12.)

---

## 10. File & directory structure

```
agent-memory-mcp/
├── DESIGN.md
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── agent-memory-mcp.toml.example
├── agent_memory_mcp/
│   ├── __init__.py            # re-exports Memory + public API
│   ├── server.py             # FastMCP app + tool adapters + main()
│   ├── memory.py             # Memory façade
│   ├── extractors.py         # RulesExtractor (default), LlmExtractor, get_extractor
│   ├── embedders.py          # HashEmbedder (default), SentenceTransformerEmbedder, get_embedder
│   ├── vector_store.py
│   ├── graph_store.py
│   ├── recall.py             # resolve/infer/traverse/vector-fallback/phrase
│   ├── models.py
│   ├── config.py
│   ├── cli.py                # Typer app
│   └── bench.py
├── seed/
│   ├── corpus.txt
│   └── questions.json
├── results/
│   └── .gitkeep
├── docs/
│   └── screenshot-placeholder.png   # referenced by README ## Screenshot
└── tests/
    ├── test_extractors.py
    ├── test_stores.py
    ├── test_traversal.py
    ├── test_recall.py
    ├── test_memory_e2e.py
    └── test_tools.py          # call the MCP tool functions directly
```

---

## 11. Build plan (ordered checklist for the builder agent)

> Complete in order; each step leaves the repo working and testable. TDD: test first where a test is named.

1. **Scaffold** the §10 tree; `pyproject.toml` (name `agent-memory-mcp`, entry point
   `agent-memory-mcp = "agent_memory_mcp.cli:app"`, deps: `mcp`, `typer`, `rich`, `numpy`, `pydantic>=2`;
   extras: `st = [sentence-transformers]`, `llm = [anthropic]`, `dev = [pytest, ruff, black]`); MIT
   `LICENSE`; `.gitignore` (ignore `results/*.json`, `*.sqlite`, `.venv`, `__pycache__`). Confirm
   `pip install -e ".[dev]"` and `agent-memory-mcp --help` (stub) run.
2. **models.py** — all §8.5 pydantic models; `load_questions(path)`, `read_corpus_lines(path)`.
3. **embedders.py** — `Embedder` protocol; `HashEmbedder` (tokenize → hash each token into `dim` buckets →
   L2-normalize); `get_embedder`; optional `SentenceTransformerEmbedder` guarded by import. Test:
   identical text → identical vector; cosine(self)=1.
4. **vector_store.py** — schema (§8.2 `facts`), `add`, `search` (numpy cosine top-k), `delete_by_entity`.
   Store float32 bytes. Test round-trip + nearest-hit.
5. **graph_store.py** — schema (`nodes`,`edges`), `add_node`/`add_edge` (idempotent upsert),
   `get_node`, `find_nodes` (case-insensitive substring/exact), `neighbors`, `traverse(start, rel, hops)`,
   `remove_entity`. Test add/get/neighbors round-trip.
6. **traversal test** — on a hand-built seed graph, `traverse` returns Farah for Dana→MANAGED_BY×2, and the
   aggregation path Search-Team→OWNS→Index-Service←USES←Atlas resolves. (`test_traversal.py`.)
7. **extractors.py** — `Extractor` protocol; `RulesExtractor`: sentence-split, for each sentence detect
   entities (proper nouns / noun phrases) and a relation from the §8.1 trigger map, emit `Statement` with
   entities+relations; unmatched → `Statement` with raw text, no relations. `get_extractor`. `LlmExtractor`
   stub raising until step 12. Test: the six §9 sentences yield the six expected relations; a garbage line
   yields zero relations but is retained as raw text.
8. **memory.py** — `Memory(db, embedder, extractor)`: `remember(text)` (split→extract→upsert nodes/edges→
   always embed raw statement→`RememberResult`); `entities`, `neighbors`, `forget`, `stats`. Node typing:
   infer `Person/Team/Service/Project` from trigger context, else `Thing`. Test `remember` populates both
   stores and is idempotent.
9. **recall.py + Memory.recall** — `resolve_entities(query)` via `find_nodes`; `infer_relation(query)` via
   trigger map; `graph_answer` (traverse from resolved entity along inferred relation(s), return terminal
   node name + path); if none, `vector_answer` (embed query, top-k, take best fact's text); `phrase` (rules:
   return the terminal name or top fact; llm optional later). Return `RecallResult` with correct `method`.
   Test: multi-hop → graph/Farah; unrelated → vector; unknown → none.
10. **cli.py** — Typer `serve`, `remember`, `recall`, `entities`, `forget`, `bench`; load `config.py`;
    ensure db dir exists; `--extractor llm` key guard. `serve` imports and runs `server.main()`.
11. **server.py** — `FastMCP("agent-memory-mcp")`; register the seven §7.1 tools as thin adapters over a
    process-lifetime `Memory` (db from config); each returns `model_dump()` dicts; `main()` runs stdio.
    `compare` calls `Memory.recall(q, method="graph")` and `Memory.recall(q, method="vector")` and packages
    both. Test (`test_tools.py`): call the tool functions directly (not over a socket) — `remember` then
    `recall`; `compare` returns both backends; `forget` removes; `stats` counts.
12. **LlmExtractor** — Anthropic structured-extraction prompt (input prose → JSON entities/relations);
    model id from env via `config.py`; single bounded retry. Unit-test with the SDK mocked (no network).
13. **bench.py** — ingest `seed/corpus.txt` via `Memory.remember`, run each question through graph-only and
    vector-only recall, grade (normalized match), build `Scorecard`, render Rich table, write JSON.
14. **End-to-end test** (`test_memory_e2e.py`) — build `Memory(":memory:")`, `remember` the seed corpus,
    `bench` in-process, assert (a) single_hop ≈ tie (both ≥ 0.8), (b) graph strictly beats vector on the
    mean of multi_hop+aggregation. Encodes the core claim; keep thresholds loose/directional.
15. **README** — pitch, the **MCP client config block** (Claude Desktop `claude_desktop_config.json` and
    Claude Code `claude mcp add`), quickstart (`pip install -e .` → `serve` / `remember` / `recall` /
    `bench`), how-it-works, going-further (st extra, `--extractor llm`), `## Screenshot` placeholder,
    clean-room note, license.
16. **Final pass** — `pytest -q` green, `ruff`/`black` clean, `serve` starts and responds to a `remember`/
    `recall` round-trip offline with defaults (no key). Capture the `bench` scorecard screenshot.
17. **Publish** — create `github.com/keithalindsay/agent-memory-mcp` (public) and push. *(Confirm the
    active `gh` account is `keithalindsay` before pushing — see prompt-regression plan's publish note.)*

---

## 12. Testing approach

- **Extraction** (`test_extractors.py`): the §9 sentences → exactly the expected relations; unmatched prose
  retained as raw (zero relations, still stored). Determinism of `RulesExtractor`.
- **Stores** (`test_stores.py`): vector add/search nearest-hit; graph node/edge round-trip; `:memory:` dbs.
- **Traversal** (`test_traversal.py`): multi-hop and aggregation resolve to the right terminal node + path.
- **Recall** (`test_recall.py`): graph path for multi-hop; vector fallback when no relation; `none` on miss;
  `method` field correct in each case.
- **Tools** (`test_tools.py`): call the FastMCP tool callables directly — remember→recall→forget→stats.
- **End-to-end** (`test_memory_e2e.py`): the graph-beats-vector claim, loose/directional thresholds.
- **Determinism:** default hash embedder + rules extractor make all non-LLM tests reproducible. The `llm`
  extractor is tested with the Anthropic SDK **mocked**; never called in CI (no key).
- Run: `pytest -q`, target < 5s.

---

## 13. Constraints & non-goals

- **IP clean-room note (REQUIRED).** This is an original, clean-room, **generic** MCP server authored by
  Keith Lindsay. It does **not** reference, reproduce, or depend on any specific employer's (including
  Aerospike's) source code, proprietary data, internal metrics, benchmarks, product names, schemas, or
  confidential specifics, and it is **not coupled to any particular graph or vector database product**. The
  only thing proprietary about the author's prior work was its integration with a specific product; the
  knowledge-graph technique itself is general, publicly-known computer science. All datasets here are
  synthetic and fictional. MIT © 2026 Keith Lindsay.
- **Robustness over rigidity.** Tools accept natural language and never require the model to produce
  structured arguments; the server tolerates unparseable input by retaining raw text for vector recall.
- **No hidden network.** The only network egress is to the selected LLM provider, and only when the `llm`
  extractor/phrasing is explicitly enabled. Default runs fully offline. No telemetry.
- **Secrets never touch disk.** `ANTHROPIC_API_KEY` is read from env only; never written to the db or logs.
- **Non-goals:** not a production/distributed memory system, not a benchmark of real KG/vector DBs, not an
  agent framework, not an entity-resolution/coreference engine. v1 value is a *usable, honest, measurable*
  memory over MCP.

---

## 14. README outline + positioning

> **Framing (required):** the README leads with the **experiment**, not the memory service. This is a
> benchmarking MCP system built to answer a concrete question — *does a knowledge graph actually beat vector
> RAG as agent memory, and where?* The memory tools exist so that comparison runs against real agent usage.

**Positioning (GitHub one-liner):** *"A benchmarking MCP server for agent memory — the same facts stored as
both vector RAG and a knowledge graph, so you can measure, live, exactly where the graph beats vectors."*

**Lead with:** (1) one-sentence statement of the experiment and why it matters, then (2) the `bench`
scorecard screenshot showing graph-vs-vector accuracy by category, then (3) the `compare` tool as the way to
run the head-to-head on any question, then (4) the MCP client config block so a reader can wire it into
Claude in one copy-paste. Emphasize: plain-language tools (nothing to format), offline/no-key default,
explainable answers with paths, and — front and center — the **measurable graph-vs-vector result**.

**Section order:** 1) Title + one-liner + badges. 2) **The experiment** (2–3 sentences: the claim, and that
this lets you test it). 3) `## Screenshot` (the `bench` scorecard). 4) `compare` in 30 seconds (a single
CLI example showing graph win vs vector miss on a multi-hop question). 5) Add to your MCP client (Claude
Desktop + Claude Code config) — use it as live memory *and* call `compare` mid-conversation. 6) Quickstart
CLI (`remember` / `recall` / `compare` / `bench`, no keys). 7) How it works (same facts → graph + vector
over one SQLite file; `method`-pinned recall; link DESIGN.md). 8) Going further (st extra; `--extractor
llm`; extend the relation vocabulary). 9) Clean-room note. 10) License.
