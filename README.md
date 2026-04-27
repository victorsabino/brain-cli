# brain — personal knowledge CLI for ~/brain.db

Single-file Python CLI that replaces every raw `sqlite3 ~/brain.db "..."` call
with typed, validated, embedded, deduplicated commands. v3 schema additive over
the original brain.db; no destructive migration.

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
| No retention/decay | recency boost (365-day half-life) + access count |
| No soft delete | `deleted_at` column |

## Install

```bash
pip3 install --break-system-packages sqlite-vec sentence-transformers
python3 scripts/migrate.py            # additive, idempotent
ln -sf $PWD/brain.py ~/bin/brain
~/bin/brain reindex                   # ~5s for 800 memories on M-series
```

## Schema additions (v3)

Aditive over the original `memories`, `memories_fts`, `stats` tables.

**New columns on `memories`**: `content_hash`, `deleted_at`, `last_accessed_at`,
`access_count`, `version`, `superseded_by`, `canonical_type`.

**New tables**:
- `tags(id, name, use_count)` — relational tag dictionary
- `memory_tags(memory_id, tag_id)` — N:M join
- `task_meta(memory_id, status, priority, energy, points, due_at, ...)` — typed task fields
- `memory_links(src_id, dst_id, kind)` — typed relations
- `memory_versions(memory_id, version, title, content, ...)` — light versioning
- `type_aliases(alias, canonical)` — type canonicalization
- `memory_vectors` (vec0 virtual) — 384-dim embeddings via sqlite-vec

**Schema version marker**: `stats.brain_schema_version = '3'`.

## Embeddings

Model: `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, ~470MB, handles PT/EN).
Storage: ~1.5KB per memory in `memory_vectors` (1.2MB for 800 entries).

Hybrid search formula:
```
final_score = (0.6 * semantic + 0.4 * BM25) * recency_boost * access_boost
recency_boost = exp(-age_days / 365)
access_boost  = 1 + log(access_count + 1)
```

## Backwards compat

The original `~/bin/brain` script (search-only) is preserved as `~/bin/brain.legacy`.
Old commands `brain query` and `brain add` still work as aliases for `search` and `save`.

## Roadmap

- v3.1 — `brain task` subcommands (done, list, due) so task ops escape raw SQL too
- v3.2 — soft archive (compress + offload memories > 2 years to cold storage)
- v3.3 — MCP server wrapper (same core, exposed as MCP tools for Claude Code)
- v3.4 — automatic relation inference (LLM proposes `caused_by` / `superseded_by` links)
