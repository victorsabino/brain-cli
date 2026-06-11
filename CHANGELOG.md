# Changelog

All notable changes to brain-cli are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2026-06-11

### Added
- `brain harvest <transcript.jsonl>` — automatic extraction from Claude Code
  session transcripts: watermark-based incremental parsing (`harvest_state`
  table, only complete new lines consumed), tool noise stripped in pure
  Python, one headless `claude -p` call (default haiku) with a REJECT-gated
  prompt, every candidate routed through the reconcile pipeline (dups and
  recall-echoes die before storage; ambiguous ones queue to
  `~/.config/brain/harvest-review.jsonl`). `--dry-run`, `--min-delta`,
  `--force`, `--model`. Recursion-guarded via `BRAIN_HARVEST=1`.
- Stop-hook template for fire-and-forget harvesting (throttled, detached) —
  documented in AGENTS.md.

## [3.3.0] - 2026-06-11

Memory-lifecycle release (schema v4): the mechanisms that keep a long-running
memory store true — informed by the documented failure modes of Mem0
(97.8%-junk audit, recall→re-extraction duplicate loops), Zep (fact
invalidation), Letta (offline consolidation), and claude-mem (token burn).

### Added
- `brain reconcile [--auto]` — ADD/UPDATE/NOOP decision packet for a candidate
  fact: top-5 neighbors with similarities; exact dups and re-extraction echoes
  (≈ memory recalled into context <24h ago) come back `noop`. The calling
  agent decides — brain itself makes no LLM calls.
- `brain invalidate <uid> [--superseded-by <uid>] [--undo]` — soft fact
  invalidation (`invalid_at`); default search/recent/context exclude
  invalidated memories, `--include-invalid` shows them, `search --as-of
  <date>` time-travels. `get` flags invalidated memories and points at the
  successor.
- `brain consolidate [--threshold] [--merge KEEPER DUP...]` — near-duplicate
  sweep (exact content hashes + embedding clusters); report mode is
  read-only, merge invalidates dups as superseded by the keeper.
- `brain context "<query>" --budget N` — prompt-ready block of L0 abstracts
  under a hard token cap; included memories are logged to `recall_log`
  (the re-extraction guard's input).
- `search --compact` — uid+title lines (~10x fewer tokens per hit).
- `--abstract` on save/update — hand-written one-line L0 summary used by
  `context` (falls back to the content head).
- Schema v4: `invalid_at`, `abstract` columns + `recall_log` table; v4
  features degrade gracefully on pre-v4 DBs until `brain migrate`.
- **Query-embedding cache** (`query_cache` table, created lazily): repeated
  search queries reuse their stored 384-dim vector and skip the embedding
  model entirely — the model is not even lazy-loaded on a cache hit
  (~27ms vs ~6.5s observed). Novel queries still pay one model load.
  Capped at 500 rows; inserting past the cap evicts the 100 oldest.
- **Opt-in ONNX embedding backend** via `BRAIN_EMBED_BACKEND=onnx`
  (needs `optimum[onnxruntime]`, deliberately NOT in the inline deps).
  Vector parity with the torch backend is exact (min cosine 1.000 over the
  `scripts/check_embed_parity.py` gate); missing extras degrade to torch
  with a `BRAIN_DEBUG` warning.
- `scripts/check_embed_parity.py` — torch-vs-ONNX parity gate; fails loudly
  (exit 1 + reindex instructions) if min cosine drops below 0.999.

## [3.2.0] - 2026-06-10

### Changed
- **Search reworked around reciprocal-rank fusion (RRF, k=60).** Keyword and
  semantic candidate lists are fused by rank, never by raw score — bm25
  magnitudes and cosine similarities no longer need a shared scale.
- FTS candidates are now **ranked by weighted bm25** (title 4×, tags 3×,
  project 2×, content 1×) instead of unweighted match order.
- Semantic retrieval moved to **chunked cosine embeddings**: content is split
  into ~500-char paragraph chunks (title prepended), one 384-dim vector per
  chunk in the `memory_chunks` vec0 table (`rowid = memory_id * 64 + chunk_index`),
  KNN takes the best chunk per memory. Long memories and appended updates stay
  findable inside the model's ~128-token window.
- FTS tokenizer switched to `porter unicode61` (stemming: "deploying" matches
  "deploy").
- `--type` / `--project` / `--since-days` filters apply **before** the
  candidate LIMIT, so filtered searches no longer starve.
- Recency (`0.05 * exp(-age/365)`) and access (`min(0.03, 0.01 * ln(1+n))`)
  are small **additive tiebreakers** on top of the fused rank score.

### Fixed
- `access_count` is bumped only on `brain get` — search no longer reinforces
  its own ranking.

### Added
- MIT license and open-source README.

## [3.1.0] - 2026-06-04

### Added
- `brain update <uid>` with `--append` (dated bullet merge) and `--reason` —
  content_hash, FTS, and embeddings stay in sync on edit.
- `alterations` audit table: every mutation logged with
  `(memory_uid, ts, kind, delta, reason)`.
- `BRAIN_DB` environment variable to point the CLI (and `scripts/migrate.py`)
  at any database file — the seam used by tests and safe mutation rehearsal.

## [3.0.0] - 2026-04-27

### Added
- Initial typed CLI over `~/brain.db` (v3 schema): `save`, `search`, `get`,
  `link`, `stats`, `recent`, `reindex`, `migrate`.
- Additive v3 schema migration: `content_hash` dedup, soft delete
  (`deleted_at`), `canonical_type` + `type_aliases`, relational tags
  (`tags` / `memory_tags`), `task_meta`, `memory_links`, `memory_versions`.
- Hybrid retrieval: FTS5 keyword + sqlite-vec semantic + recency/access decay.
- Lazy-loaded `sentence-transformers` embeddings
  (`paraphrase-multilingual-MiniLM-L12-v2`) — FTS-only fallback when the model
  is absent.
- Single-file design with uv inline script dependencies (PEP 723).

[3.4.0]: https://github.com/victorsabino/brain-cli/releases/tag/v3.4.0
[3.3.0]: https://github.com/victorsabino/brain-cli/releases/tag/v3.3.0
[3.2.0]: https://github.com/victorsabino/brain-cli/releases/tag/v3.2.0
[3.1.0]: https://github.com/victorsabino/brain-cli/releases/tag/v3.1.0
[3.0.0]: https://github.com/victorsabino/brain-cli/releases/tag/v3.0.0
