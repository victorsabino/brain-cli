# Changelog

All notable changes to brain-cli are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.5.0] - 2026-07-27

Schema v5. Closes the last open item on the June competitor-analysis roadmap
(pinned core blocks) and adds the three ideas worth taking from cognee's
memory layer — usefulness feedback, declared identity, entity anchoring —
without taking its knowledge graph or its LLM-call-per-chunk ingest.

### Added
- `brain review` — triage for the harvest review queue, which until now was
  write-only: ambiguous candidates accumulated (1098 by 2026-07-27) with no
  way to drain them, quietly making reconcile-gated capture lossy.
  `--auto` re-decides each queued candidate against the CURRENT db and
  resolves only the unambiguous ends (drops near-dups/echoes, saves clearly
  new facts), leaving genuinely ambiguous ones for a human. `--dry-run`,
  `--resolve N --action drop|save|update --uid`, `--clear --yes`, `--file`.
- `brain review --auto --judge` — a second, **orthogonal** gate. `reconcile`
  only ever measured DUPLICATION; an audit of 40 of the 780 auto-savable
  candidates found **27% transient junk** (point-in-time status, phase
  tracking, lead counts) that duplication cannot see — and best-neighbor
  similarity did NOT separate it (mean 0.546 junk vs 0.569 usable), so no
  threshold tuning could substitute. `--judge` adds one batched LLM pass
  (40 candidates/call) applying the same REJECT standard as `HARVEST_PROMPT`,
  retroactively. Junk is dropped from the ambiguous bucket too; ambiguous
  survivors stay queued, because what makes them ambiguous is duplication,
  which this pass never examined. **Fails safe**: a failed, non-zero-exit,
  or unparseable batch drops nothing, and out-of-range indices returned by
  the model are ignored — a hiccup can never silently delete facts.

  Measured on the real 1102-entry backlog (2026-07-27): dropped 270 (264 of
  them junk the judge caught), saved 621, left 211 genuinely ambiguous for a
  human. A blind 40-item re-audit of what landed put the junk rate at **10%,
  down from 27%** without the judge. Not zero — mostly time-sensitive
  prioritization notes that read as durable decisions.
- `brain feedback <uid> up|down [--note]` — explicit usefulness signal.
  `recall_log` records that a memory was *shown*; this records whether it
  *helped*. Feeds a signed, log-damped, **±0.08-capped** term in ranking:
  enough to reorder near-ties, never enough to float an irrelevant memory
  above a real lexical/semantic hit. A downvoted memory is demoted, not
  hidden (`invalidate` remains the tool for "this is wrong").
  Surfaced in `search --explain` only when feedback exists.
- `brain block set|list|rm` — pinned core blocks (roadmap item 8): a few
  short labelled values injected into every `brain context` regardless of
  query. Per-block `--char-limit`, and pins are drawn from the SAME token
  budget and hard-capped at half of it, so the feature that exists to
  prevent unbounded injection cannot become it.
- `--identity` on `save`/`reconcile` — declared merge key. An exact,
  namespaced, normalized key (`<type>:<sha256(value)[:16]>`); when a live
  memory already holds it, reconcile returns `update` **by declaration**,
  overriding similarity. Borrowed from cognee's `DataPoint`/`identity_fields`:
  a fact with no stable identity can never merge across runs. Partial-unique
  index enforces one live holder; invalidated predecessors keep theirs.
- `--anchor` on `save`/`reconcile` + anchor extraction in the harvest prompt —
  the single entity a fact is about. `brain context` groups by anchor
  (`### corena`), so recalled memory reads as knowledge about a thing rather
  than a flat list of trivia. Falls back to the flat list when unanchored.
- Schema v5: `identity_key`, `anchor` columns; `memory_feedback`, `blocks`
  tables. All v5 features degrade gracefully on pre-v5 DBs until
  `brain migrate`.
- 27 tests (`tests/test_v5.py`), fts-only.

### Changed
- `search --explain` gains `feedback_net` / `feedback_bonus`.
- `brain stats` reports `brain v5`.

### Notes
- `brain context --json` deliberately **stays a flat list**. Pinned blocks ride
  along as entries flagged `"pinned": true`, so existing consumers that ignore
  the flag see exactly what they saw before. Turning it into an object would
  have broken every agent parsing it, silently.

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
