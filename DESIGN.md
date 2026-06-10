# brain-cli — Design Notes

One page on the load-bearing decisions. If you change anything in here,
update this file in the same commit.

## Data model

```
memories (id PK, uid, title, content, canonical_type, project,
          content_hash, created_at, updated_at, deleted_at,
          last_accessed_at, access_count, version, superseded_by)
   │
   ├── memories_fts        FTS5, external-content (content=memories),
   │                       tokenize='porter unicode61'
   │                       synced by triggers memories_ai / au / ad
   │
   ├── memory_chunks       vec0 virtual table (sqlite-vec)
   │                       embedding float[384], distance_metric=cosine
   │                       rowid = memory_id * 64 + chunk_index
   │                       memory_id as metadata column
   │
   ├── memory_vectors      LEGACY vec0 (L2, one vector per memory)
   │                       read-only fallback for unmigrated DBs
   │
   ├── tags ──< memory_tags >── (N:M tag dictionary)
   ├── task_meta           typed task fields (status, priority, energy, ...)
   ├── memory_links        typed relations (src_id, dst_id, kind)
   ├── memory_versions     light versioning on update
   ├── alterations         mutation audit (memory_uid, ts, kind, delta, reason)
   └── query_cache         query-embedding cache (qhash PK, embedding BLOB,
                           created_at) — lazy-created, cap 500 / evict 100

stats.brain_schema_version = '3'
```

## Invariants (do not break)

1. **Chunk rowid scheme**: `rowid = memory_id * 64 + chunk_index`,
   `CHUNK_CAP = 64`. Rowids are deterministic, so re-embedding a memory is
   delete-range + insert with no bookkeeping table. A memory can never have
   more than 64 chunks; chunking truncates beyond the cap. Changing the cap
   renumbers every rowid → requires a full reindex.
2. **FTS sync contract**: `memories_fts` is external-content. The ONLY writers
   are the `memories_ai/au/ad` triggers. Never insert/delete on the FTS table
   directly (except the `'delete-all'`/rebuild commands inside migrations);
   bypassing triggers desyncs docids and corrupts bm25.
3. **Vector fallback matrix**:

   | memory_chunks | memory_vectors | semantic path |
   |---|---|---|
   | present | (any) | chunked KNN, cosine, best chunk per memory |
   | absent | present | legacy single-vector KNN (L2) |
   | absent | absent | none — FTS-only |

   Never write to `memory_vectors` when `memory_chunks` exists; the legacy
   table is a compatibility path for unmigrated DBs only (save falls back to
   it there), not a mirror.
4. **RRF fusion**: final score =
   `Σ 1/(k + rank) over both lists, k=60, normalized by 2/(k+1)`
   `+ 0.05 * exp(-age_days/365)` (recency, additive)
   `+ min(0.03, 0.01 * ln(1 + access_count))` (access, capped).
   Rank-based fusion is deliberate: bm25 scores and cosine distances live on
   incomparable scales; any score-mixing formula needs per-corpus calibration,
   RRF needs none. Recency/access are tiebreakers, small enough to never beat
   a genuine relevance rank difference.
5. **access_count bumps only in `cmd_get`** — never in search. Otherwise
   search reinforces its own ranking (rich-get-richer feedback loop).
6. **FTS candidates are ranked** by weighted bm25 (title 4×, tags 3×,
   project 2×, content 1×) and filters (`--type`, `--project`,
   `--since-days`) apply **before** LIMIT — both lists feeding RRF must be
   genuine top-N lists, or fusion quality silently degrades.
7. **Model-optional**: embeddings (`paraphrase-multilingual-MiniLM-L12-v2`,
   `normalize_embeddings=True`) are lazy-imported. Every command must keep
   working without sentence-transformers installed.
8. **Query cache must short-circuit the model**: `cmd_search` consults
   `query_cache` BEFORE `get_embedder()` — a hit means sentence-transformers
   is never imported in that process (~27ms repeat search vs ~6.5s cold).
   Cache writes are best-effort (`_warn` on failure), never fatal to search.
   Novel queries still pay one model load; the cache only kills repeats.
9. **Embedding backend parity**: `BRAIN_EMBED_BACKEND=onnx` may only ever
   produce vectors cosine-identical (≥ 0.999) to the torch backend —
   `scripts/check_embed_parity.py` is the gate (measured min 1.000000 exact).
   If a model/backend change breaks parity, that backend requires
   `reindex --full` before use. Missing onnx extras degrade to torch with a
   `_warn`; the extras stay OUT of the inline uv deps so the default
   `uv run brain.py` never downloads them. Measured on Apple Silicon: ONNX
   *load* is slower than torch (~8.2s vs ~4.6s warm — CoreML partitioning),
   so it's an opt-in for environments where it actually wins, not a default.

## Why single-file + uv inline deps

`brain.py` is one ~950-line module with PEP 723 inline dependencies on
purpose:

- The primary caller is an LLM agent: one file to read = full system
  context in one Read call; no import-graph spelunking.
- `uv run brain.py` resolves deps with zero project setup on any machine.
- A package split adds indirection without adding capability at this size.

`pyproject.toml` exists only for `pip install .` / entry-point installs; for
`uv run` the inline metadata wins. Keep the two dependency lists in sync.

**When to revisit**: (a) MCP server extraction — the MCP wrapper should
import brain.py's functions, which may justify a `core` split; (b) ~100k
memories — chunk KNN over ~6M rowids and full-rebuild reindex times will
need IVF/partitioning and incremental reindex, at which point the flat
module likely stops being the bottleneck worth preserving.

## Failure modes

| Condition | Behavior |
|---|---|
| sentence-transformers absent | save/search work; semantic list empty → FTS-only RRF |
| `memory_chunks` absent (pre-3.2 DB) | falls back to legacy `memory_vectors` KNN |
| both vector tables absent | keyword-only search, no error |
| duplicate `content_hash` on save | exit 2, `DUPLICATE: <uid>` (use `--force`) |
| `BRAIN_EMBED_BACKEND=onnx` w/o optimum+onnxruntime | torch backend, `_warn` |
| `query_cache` unwritable (locked/RO) | search proceeds uncached, `_warn` |
| `BRAIN_DB` set | all reads/writes (CLI + migrate) hit that path instead of `~/brain.db` |
