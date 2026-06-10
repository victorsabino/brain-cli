# brain — local memory for AI agents, in a single Python file

A persistent memory / personal knowledge base CLI for LLM coding agents
(Claude Code, Cursor, custom agents) and humans: save, search, link and dedupe learnings,
decisions, bugs, snippets and tasks in one local SQLite file. Hybrid
semantic + keyword search — porter-stemmed FTS5 plus chunked 384-dim
embeddings (sqlite-vec, cosine) fused with reciprocal rank fusion (RRF).
No server, no cloud, no SQL required: typed argparse commands over
`~/brain.db` (override with `BRAIN_DB=/path/to.db`). The v3 schema is
additive over any existing `memories` table; no destructive migration.

Works with any agent that can run a shell command: Claude Code, Cursor,
Windsurf, GitHub Copilot, OpenAI Codex CLI, Gemini CLI, Aider, Cline,
Roo Code, Continue, Goose — point it at the `brain` binary and give it
the save/search conventions in your rules file (CLAUDE.md, .cursorrules,
AGENTS.md, etc.).

```bash
brain save  --type=learning --title="..." --content="..." --tags=a,b,c
brain search "natural language query" --limit=10
brain get   <uid>
brain link  <src_uid> <dst_uid> <kind>
brain stats
brain recent 20
brain reindex     # backfill embeddings
brain migrate     # apply v3 schema
```

## What it fixes

| Old pain | v3 |
|---|---|
| Manual SQL with `''` escape | argparse handles all escaping |
| `PRAGMA trusted_schema=ON` everywhere | applied automatically by `connect()` |
| Wrong column names (`severity`?) | typed CLI args + canonical_type column |
| `tags` CSV with `LIKE '%foo%'` | normalized `tags` + `memory_tags` join table |
| Synonym expansion in every prompt | semantic search via sentence-transformers |
| Duplicate captures | content_hash dedup with explicit `--force` |
| 12 ambiguous types | 8 canonical + alias map (insight→learning, etc.) |
| No retention/decay | small additive recency/access tiebreakers (capped) |
| No soft delete | `deleted_at` column |

## Install

```bash
# via uv (recommended — deps declared inline in brain.py):
uv run brain.py migrate               # additive, idempotent
uv run brain.py reindex --full        # chunked embeddings, ~4min for 1.3K memories

# or system python:
pip3 install --break-system-packages sqlite-vec sentence-transformers
python3 scripts/migrate.py
ln -sf $PWD/brain.py ~/bin/brain
```

Point at a different DB (testing, multiple brains): `BRAIN_DB=/tmp/test.db brain ...`

## Schema additions (v3)

Additive over the original `memories`, `memories_fts`, `stats` tables.

**New columns on `memories`**: `content_hash`, `deleted_at`, `last_accessed_at`,
`access_count`, `version`, `superseded_by`, `canonical_type`.

**New tables**:
- `tags(id, name, use_count)` — relational tag dictionary
- `memory_tags(memory_id, tag_id)` — N:M join
- `task_meta(memory_id, status, priority, energy, points, due_at, ...)` — typed task fields
- `memory_links(src_id, dst_id, kind)` — typed relations
- `memory_versions(memory_id, version, title, content, ...)` — light versioning
- `type_aliases(alias, canonical)` — type canonicalization
- `memory_chunks` (vec0 virtual, cosine) — 384-dim chunked embeddings via sqlite-vec
- `memory_vectors` (vec0 virtual) — legacy single-vector table, read-only fallback
- `alterations(memory_uid, ts, kind, delta, reason)` — mutation audit trail
- `query_cache(qhash, embedding, created_at)` — query-embedding cache (lazy,
  capped at 500 rows)

**Schema version marker**: `stats.brain_schema_version = '3'`.

## Search

Two ranked candidate lists, fused by rank — never by raw score:

1. **Keyword** — FTS5 (`porter unicode61`), OR-mode terms, ordered by weighted
   bm25 (title 4×, tags 3×, project 2×, content 1×). Filters (`--type`,
   `--project`, `--since-days`) apply before the limit.
2. **Semantic** — query embedded with `paraphrase-multilingual-MiniLM-L12-v2`
   (384-dim, multilingual), KNN over `memory_chunks` (cosine), best chunk per
   memory.

```
score = RRF / (2/(k+1))            # reciprocal-rank fusion, k=60, normalized
      + 0.05 * exp(-age_days/365)  # recency: additive tiebreaker
      + min(0.03, 0.01 * ln(1 + access_count))  # access: capped tiebreaker
```

RRF is scale-free, so bm25 magnitudes and cosine similarities never need a
shared scale. `access_count` is bumped only on `brain get` — search never
reinforces its own ranking. Empty query + filters = plain listing
(`brain search "" --type=task` lists newest tasks).

Pass `--explain` to see each result's score decomposition: keyword/semantic
rank, best-chunk cosine similarity, the normalized RRF contribution of each
side, and the recency/access bonuses. In human output it's one extra line per
hit; with `--json` each result gains an `explain` object. Set `BRAIN_DEBUG=1`
to also get stage timings (model load, fts, knn, fusion, total) on stderr.

```
brain search "deploy regression" --explain
[bug     ] Rollback loop on deploy (acme)
           ...
           2026-05-01 · ab12cd34ef56 · score=1.05  #deploy
           ↳ fts=#1 sem=#2 sim=0.712 rrf=0.500+0.492 rec=+0.044 acc=+0.011 = 1.047
```

## Embeddings

Content is split into ~500-char paragraph chunks, title prepended to each, one
vector per chunk (`memory_chunks`, rowid = `memory_id * 64 + chunk_index`).
This keeps every chunk inside the model's ~128-token window, so long memories
and appended updates stay semantically findable. `save` and `update` re-embed
automatically; `reindex` backfills missing, `reindex --full` rebuilds all.

### Query-embedding cache

Search caches each query's vector in an ordinary `query_cache` table (created
lazily on first semantic search; capped at 500 rows, oldest 100 evicted on
overflow). A repeated query skips the embedding model entirely — it is not
even lazy-loaded — so repeats run in tens of milliseconds instead of paying
the ~5–10s model cold start. Novel queries still load the model once per
process.

### Optional ONNX backend

`BRAIN_EMBED_BACKEND=onnx` switches the embedder to sentence-transformers'
native ONNX backend. The extra deps are deliberately **not** in the inline uv
header (they'd bloat every run); install them per-invocation or permanently:

```bash
# one-off via uv:
BRAIN_EMBED_BACKEND=onnx uv run --with "optimum[onnxruntime]" brain.py search "..."
# or permanent: pip3 install "optimum[onnxruntime]"
```

If the extras are missing, brain falls back to the torch backend (warning
visible with `BRAIN_DEBUG=1`). Backend vector parity is gated by
`uv run scripts/check_embed_parity.py` — at min cosine ≥ 0.999 the backends
are interchangeable; below that, switching requires `reindex --full` to avoid
mixed-vector-space search. Note: on Apple Silicon the ONNX session init
(CoreML graph partitioning) can make *load* slower than torch — measure with
the parity script before adopting; the query cache is usually the bigger win.

## Backwards compat

Old commands `brain query` and `brain add` still work as aliases for `search`
and `save`. Pre-migration DBs (no `memory_chunks`) fall back to the legacy
single-vector `memory_vectors` table.

## Development

Design rationale and invariants: [DESIGN.md](DESIGN.md) · release history:
[CHANGELOG.md](CHANGELOG.md).

```bash
# Tests (smoke + update/history suites; FTS-only, no heavy deps needed)
python3 tests/test_brain.py
python3 -m pytest tests/test_update_history.py -q

# Never develop against your live ~/brain.db — use the BRAIN_DB seam:
cp ~/brain.db /tmp/dev.db
BRAIN_DB=/tmp/dev.db uv run brain.py search "deploy regression" --json
BRAIN_DB=/tmp/dev.db uv run scripts/migrate.py

# Search-quality eval (recall@1/5, MRR@10 against a golden JSONL set)
python3 scripts/eval.py --golden ~/.config/brain/golden.jsonl
python3 scripts/eval.py --golden tests/fixtures/golden_synthetic.jsonl \
    --db /tmp/fixture.db --no-semantic --min-recall5 1.0
```

`BRAIN_DB` is honored by both `brain.py` and `scripts/migrate.py`; unset, both
default to `~/brain.db`.

## Roadmap

- v3.1 — `brain task` subcommands (done, list, due) so task ops escape raw SQL too
- v3.2 — soft archive (compress + offload memories > 2 years to cold storage)
- v3.3 — MCP server wrapper (same core, exposed as MCP tools for Claude Code)
- v3.4 — automatic relation inference (LLM proposes `caused_by` / `superseded_by` links)
