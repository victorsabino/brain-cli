# brain — personal knowledge CLI for ~/brain.db

Single-file Python CLI for a local, SQLite-backed agent memory: typed,
validated, embedded, deduplicated commands over `~/brain.db` (override with
`BRAIN_DB=/path/to.db`). Hybrid retrieval — porter-stemmed FTS5 + chunked
384-dim embeddings (sqlite-vec, cosine) fused with reciprocal-rank fusion.
v3 schema is additive over any existing `memories` table; no destructive
migration. Built to be driven by LLM agents (Claude Code, etc.) without SQL.

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

## Embeddings

Content is split into ~500-char paragraph chunks, title prepended to each, one
vector per chunk (`memory_chunks`, rowid = `memory_id * 64 + chunk_index`).
This keeps every chunk inside the model's ~128-token window, so long memories
and appended updates stay semantically findable. `save` and `update` re-embed
automatically; `reindex` backfills missing, `reindex --full` rebuilds all.

## Backwards compat

Old commands `brain query` and `brain add` still work as aliases for `search`
and `save`. Pre-migration DBs (no `memory_chunks`) fall back to the legacy
single-vector `memory_vectors` table.

## Roadmap

- v3.1 — `brain task` subcommands (done, list, due) so task ops escape raw SQL too
- v3.2 — soft archive (compress + offload memories > 2 years to cold storage)
- v3.3 — MCP server wrapper (same core, exposed as MCP tools for Claude Code)
- v3.4 — automatic relation inference (LLM proposes `caused_by` / `superseded_by` links)
