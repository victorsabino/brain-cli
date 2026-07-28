#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sqlite-vec>=0.1.6",
#   "sentence-transformers>=3.0",  # only for `brain reindex` and embeddings on save
# ]
# ///
"""brain — personal knowledge CLI for ~/brain.db (v3).

Run directly with python3 if `pip install sqlite-vec` is done system-wide,
or via `uv run brain.py ...` to auto-manage all deps.

LLM-friendly: typed args via argparse, dedup, type aliases, no SQL escaping.
Search is hybrid (FTS5 keyword + sqlite-vec semantic + recency/access decay).

Heavy ML deps (sentence-transformers, torch) are LAZY-imported only when
embeddings are needed. `brain save` and `brain search` work without them
(falling back to FTS5-only retrieval).
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import secrets
import sqlite3
import sys
import time
from base64 import b32encode
from datetime import datetime, timezone
from pathlib import Path

# DB path: overridable via BRAIN_DB env var (enables safe testing against a
# throwaway copy). Defaults to ~/brain.db — unchanged behavior when unset.
DB = Path(os.environ.get("BRAIN_DB") or (Path.home() / "brain.db"))

CANONICAL_TYPES = {"learning", "decision", "bug", "snippet", "note", "task", "person", "project"}
LINK_KINDS = {"cites", "caused_by", "fixed_by", "superseded_by", "blocks", "related_to", "duplicate_of"}

# Search/embedding tuning.
RRF_K = 60          # reciprocal-rank-fusion constant (standard)
CHUNK_CAP = 64      # max chunks per memory; chunk rowid = memory_id * CHUNK_CAP + idx
CHUNK_TARGET = 500  # soft chunk size (chars) — fits the 128-token embed window
CHUNK_HARD = 900    # hard split for pathological paragraphs

# Lazy globals.
_alias_cache: dict[str, str] | None = None
_embed_model = None
_vec_loaded = False


# ────────────────────────────────────────────────────────────────────────────
# DB helpers
# ────────────────────────────────────────────────────────────────────────────


def connect(load_vec: bool = False) -> sqlite3.Connection:
    """Always opens with sane defaults. PRAGMAs that LLMs forget go here once."""
    if not DB.exists():
        sys.exit(f"✗ {DB} not found. Run `brain migrate` after creating it.")
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA trusted_schema = ON")
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL: readers never block the writer (background capture agent vs.
    # interactive command). Persists in the DB file; re-setting is a no-op.
    conn.execute("PRAGMA journal_mode = WAL")
    # Wait out short write locks instead of failing with SQLITE_BUSY.
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.row_factory = sqlite3.Row
    if load_vec:
        global _vec_loaded
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            _vec_loaded = True
        except Exception:
            _vec_loaded = False
    return conn


def _warn(msg: str) -> None:
    """Diagnostics for silently-degraded fallback paths. Off by default so
    graceful degradation stays quiet; BRAIN_DEBUG=1 surfaces them."""
    if os.environ.get("BRAIN_DEBUG") == "1":
        print(f"⚠ {msg}", file=sys.stderr)


def _timing(stage: str, start: float) -> None:
    """Stage timing to stderr, gated like _warn — zero noise by default."""
    if os.environ.get("BRAIN_DEBUG") == "1":
        print(f"⏱ {stage}: {(time.perf_counter() - start) * 1000:.1f}ms", file=sys.stderr)


def aliases(conn: sqlite3.Connection) -> dict[str, str]:
    global _alias_cache
    if _alias_cache is not None:
        return _alias_cache
    rows = conn.execute("SELECT alias, canonical FROM type_aliases").fetchall()
    _alias_cache = {r["alias"]: r["canonical"] for r in rows}
    return _alias_cache


def canonical_type(conn: sqlite3.Connection, raw: str) -> str:
    return aliases(conn).get((raw or "").lower().strip(), "note")


def gen_uid() -> str:
    """12-char base32 (~60 bits, collision risk ~1B entries). Lowercase, no padding."""
    return b32encode(secrets.token_bytes(8))[:12].decode("ascii").lower()


def content_hash(title: str, content: str) -> str:
    return hashlib.sha256(f"{title}\n{content or ''}".encode()).hexdigest()[:16]


_flags_cache: dict[str, bool] | None = None


def _schema_flags(conn: sqlite3.Connection) -> dict[str, bool]:
    """Which v4/v5 columns exist. Lets new code run against un-migrated DBs
    (features quietly degrade instead of erroring on missing columns)."""
    global _flags_cache
    if _flags_cache is None:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
        _flags_cache = {
            "invalid_at": "invalid_at" in cols,
            "abstract": "abstract" in cols,
            "identity_key": "identity_key" in cols,   # v5
            "anchor": "anchor" in cols,               # v5
        }
    return _flags_cache


# ────────────────────────────────────────────────────────────────────────────
# v5: identity keys, feedback-weighted recall, pinned blocks
# ────────────────────────────────────────────────────────────────────────────

# Explicit-identity dedup. Borrowed from cognee's DataPoint: a node with a
# random id can never merge across runs, so mergeable facts declare the fields
# that define them and the key is derived deterministically from those.
# Exact key, never a shared token or substring — a generic token betrays you
# ("health" matching sutterhealth, "valley" matching two unrelated orgs).
def identity_key(kind: str, value: str) -> str:
    """Stable key for a fact that should merge instead of duplicating.

    `kind` namespaces the value so two different fact-kinds sharing a value
    (project "corena" vs person "corena") never collide.
    """
    norm = " ".join((value or "").split()).strip().lower()
    if not norm:
        return ""
    return f"{kind.strip().lower()}:{hashlib.sha256(norm.encode()).hexdigest()[:16]}"


def ensure_feedback_table(conn: sqlite3.Connection) -> None:
    """Idempotent. Explicit usefulness signal per memory.

    recall_log says a memory was *shown*; that is not the same as useful.
    Cognee reweights graph edges from answer feedback — this is the flat-store
    equivalent: an agent or human marks a recalled memory as having helped (or
    misled), and ranking reflects it.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_feedback (
            id        INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL,
            signal    INTEGER NOT NULL,      -- +1 useful, -1 misleading
            note      TEXT,
            ts        TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback(memory_id)"
    )


def ensure_blocks_table(conn: sqlite3.Connection) -> None:
    """Idempotent. Pinned core blocks — tiny always-injected context.

    Roadmap item 8. A handful of short labelled values (persona, hard rules,
    current focus) that should ride along with EVERY context block regardless
    of what the query matched. Hard char_limit per block so this can never
    become the unbounded-injection problem it exists to avoid.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blocks (
            label      TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            char_limit INTEGER NOT NULL DEFAULT 400,
            position   INTEGER NOT NULL DEFAULT 100,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)


def _feedback_scores(conn: sqlite3.Connection, memory_ids: list[int]) -> dict[int, float]:
    """Net feedback per memory, damped. Empty dict if the table does not exist
    yet (pre-v5 DB) so ranking degrades to exactly the v4 behaviour."""
    if not memory_ids:
        return {}
    try:
        ph = ",".join("?" * len(memory_ids))
        return {
            r[0]: float(r[1] or 0)
            for r in conn.execute(
                f"SELECT memory_id, SUM(signal) FROM memory_feedback "
                f"WHERE memory_id IN ({ph}) GROUP BY memory_id", memory_ids)
        }
    except sqlite3.OperationalError:
        return {}


def _log_recall(conn: sqlite3.Connection, memory_ids: list[int]) -> None:
    """Record that these memories were surfaced into an agent's context.

    `brain reconcile` reads this as a re-extraction guard: a candidate fact
    that closely matches a recently-recalled memory is almost always the
    agent re-saving its own injected context, not new knowledge.
    Best-effort — recall must never fail because logging did.
    """
    if not memory_ids:
        return
    try:
        conn.executemany(
            "INSERT INTO recall_log(memory_id) VALUES (?)",
            [(m,) for m in memory_ids],
        )
        conn.commit()
    except sqlite3.OperationalError as e:
        _warn(f"recall_log write skipped ({e})")


def _recently_recalled(conn: sqlite3.Connection, hours: int = 24) -> set[int]:
    try:
        return {r[0] for r in conn.execute(
            "SELECT DISTINCT memory_id FROM recall_log WHERE recalled_at > datetime('now', ?)",
            (f"-{hours} hours",),
        )}
    except sqlite3.OperationalError:
        return set()


def _auto_abstract(content: str, max_chars: int = 240) -> str:
    """L0 fallback when no explicit --abstract was saved: head of the content,
    whitespace-collapsed. Good enough for context lines; explicit abstracts
    written by the saving agent are better."""
    text = " ".join((content or "").split())
    return text[:max_chars] + ("…" if len(text) > max_chars else "")


# ────────────────────────────────────────────────────────────────────────────
# Alterations (revision audit trail)
# ────────────────────────────────────────────────────────────────────────────


def ensure_alterations_table(conn: sqlite3.Connection) -> None:
    """Idempotent. Tracks every mutation (create/append/replace/delete) to a memory.

    Lazy — called from the commands that write alterations so we never force a
    full re-migrate. Also created by `brain migrate`.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alterations (
            id          INTEGER PRIMARY KEY,
            memory_uid  TEXT NOT NULL,
            ts          TEXT NOT NULL,
            kind        TEXT NOT NULL,
            delta       TEXT,
            reason      TEXT
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_alterations_uid ON alterations(memory_uid)"
    )


def log_alteration(conn, uid: str, kind: str, delta: str | None = None,
                   reason: str | None = None) -> None:
    """Insert one alterations row. Caller commits."""
    ensure_alterations_table(conn)
    conn.execute(
        "INSERT INTO alterations(memory_uid, ts, kind, delta, reason) "
        "VALUES (?, datetime('now'), ?, ?, ?)",
        (uid, kind, delta, reason),
    )


# ────────────────────────────────────────────────────────────────────────────
# Embeddings (lazy)
# ────────────────────────────────────────────────────────────────────────────


def get_embedder():
    """Lazy-load sentence-transformers. Returns None if not installed.

    BRAIN_EMBED_BACKEND=onnx opts into the ONNX runtime (much faster model
    load than torch; needs optimum+onnxruntime — NOT in the inline deps on
    purpose, see README). Missing extras or an old sentence-transformers
    (<3.2, no `backend` kwarg) degrade to the torch backend with a _warn.
    Backend parity is gated by scripts/check_embed_parity.py — if min cosine
    between backends ever drops below 0.999, switching requires
    `reindex --full` to avoid mixed-vector-space search.
    """
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    # Multilingual, 384-dim, ~470MB, handles PT/EN well.
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    if os.environ.get("BRAIN_EMBED_BACKEND", "").lower() == "onnx":
        try:
            import onnxruntime  # noqa: F401 — presence check only
            import optimum  # noqa: F401
            _embed_model = SentenceTransformer(model_name, backend="onnx")
            return _embed_model
        except ImportError:
            _warn("BRAIN_EMBED_BACKEND=onnx but optimum/onnxruntime missing; "
                  "falling back to torch backend")
        except (TypeError, ValueError, OSError) as e:
            _warn(f"onnx backend failed ({e}); falling back to torch backend")
    _embed_model = SentenceTransformer(model_name)
    return _embed_model


def embed(text: str) -> bytes | None:
    model = get_embedder()
    if model is None:
        return None
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype("float32").tobytes()


# Query-embedding cache: repeated searches skip the model load entirely
# (agents re-run the same queries a lot). Novel queries still pay the load.
QUERY_CACHE_MAX = 500    # rows before eviction kicks in
QUERY_CACHE_EVICT = 100  # oldest rows dropped per eviction


def _query_cache_get(conn: sqlite3.Connection, qhash: str) -> bytes | None:
    """Cached query vector or None. Missing table / read errors → None."""
    try:
        r = conn.execute(
            "SELECT embedding FROM query_cache WHERE qhash = ?", (qhash,)
        ).fetchone()
        return r["embedding"] if r else None
    except sqlite3.OperationalError:
        return None  # table not created yet — first cacheable search makes it


def _query_cache_put(conn: sqlite3.Connection, qhash: str, blob: bytes) -> None:
    """Insert (lazily creating the table), evict oldest when over cap.
    Best-effort: a locked/read-only DB must never break search itself."""
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                qhash      TEXT PRIMARY KEY,
                embedding  BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        conn.execute(
            "INSERT OR REPLACE INTO query_cache(qhash, embedding) VALUES (?, ?)",
            (qhash, blob),
        )
        n = conn.execute("SELECT COUNT(*) AS n FROM query_cache").fetchone()["n"]
        if n > QUERY_CACHE_MAX:
            conn.execute(f"""
                DELETE FROM query_cache WHERE qhash IN (
                    SELECT qhash FROM query_cache
                    ORDER BY created_at ASC, rowid ASC LIMIT {QUERY_CACHE_EVICT}
                )
            """)
        conn.commit()
    except sqlite3.OperationalError as e:
        _warn(f"query_cache write skipped ({e})")


def _split_chunks(content: str) -> list[str]:
    """Split content into ~CHUNK_TARGET-char pieces on paragraph boundaries."""
    paras: list[str] = []
    for p in re.split(r"\n\s*\n", content or ""):
        p = p.strip()
        while len(p) > CHUNK_HARD:
            paras.append(p[:CHUNK_HARD])
            p = p[CHUNK_HARD:].strip()
        if p:
            paras.append(p)
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > CHUNK_TARGET:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks or [""]


def chunk_texts(title: str, content: str, anchor: str | None = None) -> list[str]:
    """Embeddable texts: title (and anchor) prepended to every chunk so each
    stays topical.

    v6: the anchor is the entity the fact is about. Without it in the embedded
    text, a query that describes the situation but not the proper noun has
    nothing to latch onto and sibling memories in the same cluster win — the
    dominant failure mode in the 2026-07-27 reachability audit (vague-phrasing
    recall@5 was 0.62 against 0.98 for keyword queries).
    """
    head = f"{anchor}\n{title}" if anchor else title
    return [f"{head}\n{c}" if c else head for c in _split_chunks(content)][:CHUNK_CAP]


def embed_memory(conn: sqlite3.Connection, memory_id: int, title: str, content: str,
                 anchor: str | None = None) -> bool:
    """Write chunked embeddings into memory_chunks (cosine vec0).

    The model's effective window is ~128 tokens, so one vector per memory makes
    anything past the first ~90 words (and every `update --append`) invisible
    to semantic search. Chunking fixes that. Falls back to the legacy
    single-vector memory_vectors table if memory_chunks doesn't exist yet.

    Does NOT commit — the caller owns the transaction, so the memory row and
    its chunks land atomically (a crash between two separate commits used to
    leave memories silently unsearchable; `brain doctor` finds survivors).
    """
    model = get_embedder()
    if model is None or not _vec_loaded:
        return False
    texts = chunk_texts(title, content, anchor)
    try:
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        base = memory_id * CHUNK_CAP
        for i in range(CHUNK_CAP):
            conn.execute("DELETE FROM memory_chunks WHERE rowid = ?", (base + i,))
        for i, v in enumerate(vecs):
            conn.execute(
                "INSERT INTO memory_chunks(rowid, memory_id, embedding) VALUES (?, ?, ?)",
                (base + i, memory_id, v.astype("float32").tobytes()),
            )
        return True
    except sqlite3.OperationalError as e:
        _warn(f"memory_chunks write failed ({e}); trying legacy memory_vectors")
    # Legacy fallback (pre-migration DB): single vector, L2 table.
    vec = embed(f"{title}\n{content}")
    if vec is None:
        return False
    try:
        conn.execute(
            "INSERT OR REPLACE INTO memory_vectors(memory_id, embedding) VALUES (?, ?)",
            (memory_id, vec),
        )
        return True
    except sqlite3.OperationalError as e:
        _warn(f"memory_vectors write failed: {e}")
        return False


def _embed_in_txn(conn, cursor, memory_id: int, title: str, content: str, uid: str,
                  anchor: str | None = None) -> None:
    """Embed inside the caller's OPEN transaction (savepoint-guarded).

    Atomicity: memory row + chunks commit together — no crash window where a
    committed memory has no vectors. Resilience: an embedding exception rolls
    back only the savepoint, so the save itself still commits (FTS-only mode).
    """
    cursor.execute("SAVEPOINT embed")
    try:
        embed_memory(conn, memory_id, title, content, anchor)
        cursor.execute("RELEASE SAVEPOINT embed")
    except Exception as e:
        cursor.execute("ROLLBACK TO SAVEPOINT embed")
        cursor.execute("RELEASE SAVEPOINT embed")
        print(f"⚠ embedding failed for {uid} ({e}); saved FTS-only — `brain doctor --fix` repairs",
              file=sys.stderr)


# ────────────────────────────────────────────────────────────────────────────
# Commands
# ────────────────────────────────────────────────────────────────────────────


def cmd_save(args) -> int:
    conn = connect(load_vec=True)
    cursor = conn.cursor()

    raw_type = args.type
    ctype = canonical_type(conn, raw_type)
    title = args.title.strip()
    content = (args.content or "").strip()

    if not title:
        return _err("--title is required and must not be empty")
    if ctype not in CANONICAL_TYPES:
        return _err(f"unknown type '{raw_type}' (canonical: {ctype}). Valid: {sorted(CANONICAL_TYPES)}")

    h = content_hash(title, content)

    # Dedup
    existing = cursor.execute(
        "SELECT uid FROM memories WHERE content_hash = ? AND deleted_at IS NULL LIMIT 1",
        (h,),
    ).fetchone()
    if existing and not args.force:
        print(f"DUPLICATE: {existing['uid']}", file=sys.stderr)
        print(existing["uid"])
        return 2

    uid = gen_uid()
    cursor.execute("""
        INSERT INTO memories (uid, type, canonical_type, title, content, project, area,
                              tags, source_file, content_hash, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
    """, (
        uid, raw_type.lower().strip(), ctype, title, content,
        args.project or None, args.area or None,
        ",".join(args.tags) if args.tags else None,
        args.source_file or None,
        h,
    ))
    memory_id = cursor.lastrowid

    # Tags into normalized table.
    for raw_tag in (args.tags or []):
        for tag in raw_tag.split(","):
            tag = tag.strip().lower()
            if not tag:
                continue
            cursor.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag,))
            tag_id = cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()["id"]
            try:
                cursor.execute("INSERT INTO memory_tags(memory_id, tag_id) VALUES (?, ?)",
                               (memory_id, tag_id))
            except sqlite3.IntegrityError:
                pass

    # Task-specific metadata.
    if ctype == "task":
        cursor.execute("""
            INSERT INTO task_meta(memory_id, status, priority, energy, points, due_at, external_ref, parent_uid)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, args.status or "pending", args.priority, args.energy, args.points,
              args.due_at, args.external_ref, args.parent_uid))

    # L0 abstract (one searchable sentence for `brain context` injection).
    if getattr(args, "abstract", None):
        if _schema_flags(conn)["abstract"]:
            cursor.execute("UPDATE memories SET abstract = ? WHERE id = ?",
                           (args.abstract.strip(), memory_id))
        else:
            _warn("--abstract ignored — run `brain migrate` (schema v4)")

    # v5: declared identity (deterministic merge key) and anchor (the entity a
    # lesson is about). Both no-op on a pre-v5 DB.
    flags = _schema_flags(conn)
    ident = getattr(args, "identity", None)
    if ident:
        if flags.get("identity_key"):
            cursor.execute("UPDATE memories SET identity_key = ? WHERE id = ?",
                           (identity_key(ctype, ident), memory_id))
        else:
            _warn("--identity ignored — run `brain migrate` (schema v5)")
    anchor = getattr(args, "anchor", None)
    if anchor:
        if flags.get("anchor"):
            cursor.execute("UPDATE memories SET anchor = ? WHERE id = ?",
                           (anchor.strip(), memory_id))
        else:
            _warn("--anchor ignored — run `brain migrate` (schema v5)")

    # Audit trail: record creation.
    log_alteration(conn, uid, "create", delta=title, reason=None)

    # Embed in the SAME transaction, then one commit (see _embed_in_txn).
    if not args.no_embed and _vec_loaded:
        _embed_in_txn(conn, cursor, memory_id, title, content, uid,
                      anchor.strip() if anchor else None)

    conn.commit()

    print(uid)
    return 0


def _build_filters(conn: sqlite3.Connection, args) -> tuple[str, list]:
    """Shared --type/--project/--since-days + validity filter SQL (alias `m`).

    Validity: invalidated facts (invalid_at set) are excluded by default —
    stale knowledge must not outrank live knowledge. `--include-invalid`
    shows everything; `--as-of <date>` time-travels (what was true then).
    """
    sql, params = "", []
    if getattr(args, "type", None):
        sql += f" AND m.canonical_type IN ({','.join('?' * len(args.type))})"
        params.extend(canonical_type(conn, t) for t in args.type)
    if getattr(args, "project", None):
        sql += " AND m.project = ?"
        params.append(args.project)
    if getattr(args, "since_days", None):
        sql += " AND m.created_at > datetime('now', ?)"
        params.append(f"-{args.since_days} days")
    if _schema_flags(conn)["invalid_at"]:
        as_of = getattr(args, "as_of", None)
        if as_of:
            sql += " AND m.created_at <= ? AND (m.invalid_at IS NULL OR m.invalid_at > ?)"
            params.extend([as_of, as_of])
        elif not getattr(args, "include_invalid", False):
            sql += " AND m.invalid_at IS NULL"
    return sql, params


def _abstract_col(conn: sqlite3.Connection) -> str:
    return "m.abstract" if _schema_flags(conn)["abstract"] else "NULL AS abstract"


def _anchor_col(conn: sqlite3.Connection) -> str:
    return "m.anchor" if _schema_flags(conn).get("anchor") else "NULL AS anchor"


def _row_get(row, key: str, default=None):
    """sqlite3.Row has no .get(); columns also vary by schema version."""
    try:
        v = row[key]
    except (IndexError, KeyError):
        return default
    return default if v is None else v


def _hybrid_search(conn, cursor, query: str, limit: int, filter_sql: str,
                   filter_params: list, *, no_semantic: bool = False,
                   want_explain: bool = False):
    """Retrieval core shared by search / context / reconcile.

    Ranked FTS5 + chunked semantic KNN fused with normalized RRF, plus small
    additive recency/access bonuses. Returns (top, explain) where top is
    [(score, row)] and explain maps memory id → score decomposition (always
    populated when want_explain — reconcile needs the best-chunk sims).
    """
    pool = max(50, limit * 5)

    # 1. FTS5 keyword — best-first (bm25 ASC), title/tags weighted above body.
    #    Filters applied BEFORE the limit so filtered searches don't starve.
    fts_ids: list[int] = []
    words = [w.replace('"', "") for w in query.split()]
    fts_terms = " OR ".join(f'"{w}"' for w in words if len(w) > 1)
    if fts_terms:
        t_fts = time.perf_counter()
        try:
            fts_ids = [r["id"] for r in cursor.execute(f"""
                SELECT m.id AS id
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ? AND m.deleted_at IS NULL{filter_sql}
                ORDER BY bm25(memories_fts, 4.0, 1.0, 3.0, 2.0, 1.0)
                LIMIT ?
            """, (fts_terms, *filter_params, pool))]
        except sqlite3.OperationalError as e:
            _warn(f"FTS query failed ({e}); keyword side skipped")
        _timing("fts query", t_fts)

    # 2. Semantic — chunked KNN (cosine), best chunk per memory. Oversampled
    #    because several chunks can belong to one memory and filters apply after.
    sem_pairs: list[tuple[int, float]] = []
    if _vec_loaded and not no_semantic:
        # Query-embedding cache first: a hit means the model is never even
        # lazy-loaded — repeat queries go straight to KNN.
        qhash = hashlib.sha256(query.encode()).hexdigest()
        t_cache = time.perf_counter()
        qvec = _query_cache_get(conn, qhash)
        _timing("query cache " + ("hit" if qvec is not None else "miss"), t_cache)
        if qvec is None:
            t_model = time.perf_counter()
            get_embedder()  # idempotent — surface the lazy model-load cost alone
            _timing("model load", t_model)
            qvec = embed(query)
            if qvec is not None:
                _query_cache_put(conn, qhash, qvec)
        t_knn = time.perf_counter()
        if qvec is not None:
            k = pool * (4 if filter_sql else 2)
            best: dict[int, float] = {}
            try:
                for r in cursor.execute("""
                    SELECT memory_id AS id, distance
                    FROM memory_chunks
                    WHERE embedding MATCH ? AND k = ?
                """, (qvec, k)):
                    sim = 1.0 - r["distance"]  # cosine distance → similarity
                    if sim > best.get(r["id"], -1.0):
                        best[r["id"]] = sim
            except sqlite3.OperationalError as e:
                _warn(f"memory_chunks KNN failed ({e}); trying legacy memory_vectors")
                # Legacy single-vector table: L2 over normalized vecs → cos = 1 - d²/2.
                try:
                    for r in cursor.execute("""
                        SELECT memory_id AS id, distance
                        FROM memory_vectors
                        WHERE embedding MATCH ? AND k = ?
                    """, (qvec, k)):
                        best[r["id"]] = 1.0 - (r["distance"] ** 2) / 2.0
                except sqlite3.OperationalError as e2:
                    _warn(f"memory_vectors KNN failed ({e2}); semantic side skipped")
            sem_pairs = sorted(best.items(), key=lambda x: x[1], reverse=True)
        _timing("knn query", t_knn)

    candidate_ids = set(fts_ids) | {mid for mid, _ in sem_pairs}
    if not candidate_ids:
        return [], ({} if want_explain else None)

    placeholders = ",".join("?" * len(candidate_ids))
    rows = cursor.execute(f"""
        SELECT m.id, m.uid, m.canonical_type AS type, m.title, m.content, m.project,
               m.created_at, m.access_count, {_abstract_col(conn)}, {_anchor_col(conn)}
        FROM memories m
        WHERE m.id IN ({placeholders}) AND m.deleted_at IS NULL{filter_sql}
    """, (*candidate_ids, *filter_params)).fetchall()
    by_id = {r["id"]: r for r in rows}

    # Ranks among surviving (filter-passing) candidates only.
    fts_rank = {mid: i for i, mid in enumerate(m for m in fts_ids if m in by_id)}
    sem_rank = {mid: i for i, mid in enumerate(mid for mid, _ in sem_pairs if mid in by_id)}

    # 3. RRF fusion, normalized so rank-1 in both lists ≈ 1.0, then small
    #    additive recency/access bonuses (tiebreakers, capped).
    t_fusion = time.perf_counter()
    sims = dict(sem_pairs)  # best-chunk cosine sim per memory
    explain: dict[int, dict] | None = {} if want_explain else None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    full = 2.0 / (RRF_K + 1)
    fb = _feedback_scores(conn, list(by_id.keys()))
    scored = []
    for mid, r in by_id.items():
        rrf_fts = 1.0 / (RRF_K + fts_rank[mid] + 1) / full if mid in fts_rank else 0.0
        rrf_sem = 1.0 / (RRF_K + sem_rank[mid] + 1) / full if mid in sem_rank else 0.0
        try:
            age_days = (now - datetime.fromisoformat(str(r["created_at"] or "").split(".")[0])).days
        except (TypeError, ValueError):
            age_days = 9999
        recency = 0.05 * math.exp(-max(age_days, 0) / 365.0)
        access = min(0.03, 0.01 * math.log1p(r["access_count"] or 0))
        # Feedback: signed, log-damped, capped at ±0.08 — big enough to reorder
        # near-ties, never big enough to float an irrelevant memory above a
        # genuine lexical/semantic hit. A downvoted memory is demoted, not
        # hidden; invalidate is the tool for "this is wrong".
        net = fb.get(mid, 0.0)
        useful = math.copysign(min(0.08, 0.04 * math.log1p(abs(net))), net) if net else 0.0
        score = rrf_fts + rrf_sem + recency + access + useful
        if explain is not None:
            explain[mid] = {
                "fts_rank": fts_rank.get(mid),
                "sem_rank": sem_rank.get(mid),
                "sim": round(sims[mid], 4) if mid in sims else None,
                "rrf_fts": round(rrf_fts, 4), "rrf_sem": round(rrf_sem, 4),
                "rrf": round(rrf_fts + rrf_sem, 4),
                "recency_bonus": round(recency, 4), "access_bonus": round(access, 4),
                "feedback_net": net, "feedback_bonus": round(useful, 4),
                "final": round(score, 4),
            }
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    _timing("fusion", t_fusion)
    return scored[:limit], explain


def _explain_line(e: dict) -> str:
    """One compact human line: ranks, best-chunk sim, rrf parts, bonuses."""
    fts = f"#{e['fts_rank'] + 1}" if e["fts_rank"] is not None else "-"
    sem = f"#{e['sem_rank'] + 1}" if e["sem_rank"] is not None else "-"
    sim = f"{e['sim']:.3f}" if e["sim"] is not None else "-"
    # only show the feedback term when there IS feedback — keeps the common
    # line unchanged and makes a reordering caused by feedback obvious
    fbs = ""
    if e.get("feedback_net"):
        fbs = f"fb={e['feedback_net']:+g}({e['feedback_bonus']:+.3f}) "
    return (f"fts={fts} sem={sem} sim={sim} "
            f"rrf={e['rrf_fts']:.3f}+{e['rrf_sem']:.3f} "
            f"rec=+{e['recency_bonus']:.3f} acc=+{e['access_bonus']:.3f} "
            f"{fbs}= {e['final']:.3f}")


def _print_results(top: list, conn: sqlite3.Connection, query: str, as_json: bool,
                   explain: dict[int, dict] | None = None, compact: bool = False) -> None:
    if as_json:
        out = []
        for score, r in top:
            if compact:
                d = {"uid": r["uid"], "type": r["type"], "title": r["title"],
                     "created_at": r["created_at"], "score": round(score, 3)}
            else:
                d = {
                    "uid": r["uid"], "type": r["type"], "title": r["title"],
                    "snippet": (r["content"] or "")[:300],
                    "project": r["project"], "score": round(score, 3),
                    "created_at": r["created_at"],
                }
            if explain is not None and r["id"] in explain:
                d["explain"] = explain[r["id"]]
            out.append(d)
        print(json.dumps(out, ensure_ascii=False, indent=2))
    elif compact:
        # ~10x fewer tokens per hit: uid + title only. Agents filter here,
        # then `brain get` the few that matter (claude-mem's 3-step lesson).
        for score, r in top:
            print(f"{r['uid']}  [{r['type']:8}] {r['title']}  ({str(r['created_at'] or '')[:10]})")
    else:
        print(f"\n{len(top)} hit(s) for: {query}\n")
        for score, r in top:
            tags = _tags_for(conn, r["id"])
            tag_str = f"  #{','.join(tags)}" if tags else ""
            project = f" ({r['project']})" if r["project"] else ""
            snippet = (r["content"] or "").replace("\n", " ")[:150]
            date = (r["created_at"] or "")[:10]
            print(f"[{r['type']:8}] {r['title']}{project}")
            if snippet:
                print(f"           {snippet}{'…' if len(r['content'] or '') > 150 else ''}")
            print(f"           {date} · {r['uid']} · score={score:.2f}{tag_str}")
            if explain is not None and r["id"] in explain:
                print(f"           ↳ {_explain_line(explain[r['id']])}")
            print()


def cmd_search(args) -> int:
    """Hybrid search: ranked FTS5 + chunked semantic KNN, fused with RRF.

    RRF is scale-free — it merges by *rank*, so bm25 magnitudes and cosine
    similarities never have to share a scale (the old weighted-sum did, and
    three normalization bugs buried good hits). Recency/access are small
    additive tiebreakers, never multiplicative gates.
    """
    t_total = time.perf_counter()
    conn = connect(load_vec=True)
    cursor = conn.cursor()

    query = (args.query or "").strip()
    limit = args.limit
    filter_sql, filter_params = _build_filters(conn, args)

    # Empty query → plain filtered listing (newest first). Makes the
    # documented `brain search "" --type=task` pattern actually work.
    if not query:
        rows = cursor.execute(f"""
            SELECT m.id, m.uid, m.canonical_type AS type, m.title, m.content, m.project,
                   m.created_at, m.access_count
            FROM memories m
            WHERE m.deleted_at IS NULL{filter_sql}
            ORDER BY m.updated_at DESC
            LIMIT ?
        """, (*filter_params, limit)).fetchall()
        _print_results([(0.0, r) for r in rows], conn, "(listing)", args.json,
                       compact=getattr(args, "compact", False))
        return 0

    top, explain = _hybrid_search(
        conn, cursor, query, limit, filter_sql, filter_params,
        no_semantic=args.no_semantic, want_explain=getattr(args, "explain", False),
    )
    _timing("total", t_total)
    if not top:
        print("[]" if args.json else f"No results for: {query}")
        return 0

    # access_count is bumped on `brain get` only — search must not reinforce
    # its own ranking (rich-get-richer loop on frequently-surfaced junk).
    _print_results(top, conn, query, args.json, explain=explain,
                   compact=getattr(args, "compact", False))
    return 0


def cmd_context(args) -> int:
    """Emit a prompt-ready block of relevant memories under a hard token budget.

    L0 abstracts only (~1 line per memory) — agents inject this at session
    start / prompt time, then `brain get <uid>` for anything that matters.
    Budget is a HARD cap (≈4 chars/token); claude-mem's top complaint class
    is unbounded memory injection burning the user's context window.
    Included memories are recorded in recall_log (re-extraction guard).
    """
    conn = connect(load_vec=True)
    cursor = conn.cursor()
    query = (args.query or "").strip()
    if not query:
        return _err("query required (e.g. the project name or current task)")

    filter_sql, filter_params = _build_filters(conn, args)
    top, _ = _hybrid_search(conn, cursor, query, args.limit, filter_sql,
                            filter_params, no_semantic=args.no_semantic)
    if not top:
        print(f"(no brain memories matched: {query})")
        return 0

    budget_chars = max(args.budget, 100) * 4
    header = f"## Relevant memories (brain) — {query}"
    footer = "(full detail: `brain get <uid>`)"
    # Pinned blocks come out of the SAME budget — they are always-injected, so
    # if they were free they would be the unbounded-injection problem wearing a
    # different hat. They win ties against matched memories (that is the point
    # of pinning) but they cannot silently double the block size.
    pinned = [] if getattr(args, "no_blocks", False) else _pinned_blocks(conn)
    pin_lines = [f"- [{r['label']}] {r['value'][:r['char_limit']]}" for r in pinned]
    pin_chars = sum(len(x) + 1 for x in pin_lines)
    if pin_chars > budget_chars // 2 and pin_lines:
        # never let pins eat more than half the budget
        kept, acc = [], 0
        for line in pin_lines:
            if acc + len(line) + 1 > budget_chars // 2:
                break
            kept.append(line)
            acc += len(line) + 1
        _warn(f"pinned blocks trimmed {len(pin_lines)}→{len(kept)} to stay under half the budget")
        pin_lines, pin_chars = kept, acc
    used = len(header) + len(footer) + 2 + pin_chars
    entries = []
    for score, r in top:
        abstract = r["abstract"] or _auto_abstract(r["content"])
        line = f"- [{r['type']}] {r['title']} — {abstract} ({r['uid']}, {str(r['created_at'] or '')[:10]})"
        if entries and used + len(line) + 1 > budget_chars:
            break
        entries.append((line, r, abstract, score))
        used += len(line) + 1

    if args.json:
        # Stays a flat LIST — agents already parse this shape, and turning it
        # into an object would break them silently. Pinned blocks ride along as
        # entries flagged `pinned`, so a consumer that ignores the flag sees
        # exactly what it saw before.
        out = [{
            "uid": None, "type": "block", "title": r["label"],
            "abstract": r["value"][:r["char_limit"]], "project": None,
            "created_at": None, "score": None, "pinned": True,
        } for r in pinned[:len(pin_lines)]]
        out += [{
            "uid": r["uid"], "type": r["type"], "title": r["title"],
            "abstract": abstract, "project": r["project"],
            "anchor": _row_get(r, "anchor"),
            "created_at": r["created_at"], "score": round(score, 3),
        } for _, r, abstract, score in entries]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if pin_lines:
            print("## Pinned (brain)")
            for line in pin_lines:
                print(line)
            print()
        print(header)
        # Group by anchor when the DB has them: entity-anchored lessons read as
        # knowledge about a thing, where a flat list reads as trivia. Falls back
        # to the flat list when nothing is anchored (pre-v5 or unanchored data).
        anchored = [(line, r) for line, r, _, _ in entries
                    if _schema_flags(conn).get("anchor") and _row_get(r, "anchor")]
        if anchored and len(anchored) >= 2:
            groups: dict[str, list[str]] = {}
            for line, r in anchored:
                groups.setdefault(_row_get(r, "anchor"), []).append(line)
            done = {id(line) for line, _ in anchored}
            for anchor in sorted(groups):
                print(f"### {anchor}")
                for line in groups[anchor]:
                    print(line)
            rest = [line for line, *_ in entries if id(line) not in done]
            if rest:
                print("### other")
                for line in rest:
                    print(line)
        else:
            for line, *_ in entries:
                print(line)
        print(footer)

    _log_recall(conn, [r["id"] for _, r, _, _ in entries])
    return 0


def cmd_feedback(args) -> int:
    """Mark a recalled memory as useful (+1) or misleading (-1).

    This is the signal cognee gets from answer ratings and uses to reweight
    graph edges; here it is a small, capped additive term in ranking. Kept
    deliberately blunt — one integer per event, no decay model — because the
    interesting question is whether the signal helps at all, and a simple
    counter is auditable.
    """
    conn = connect()
    ensure_feedback_table(conn)
    row = conn.execute(
        "SELECT id, title FROM memories WHERE uid = ? AND deleted_at IS NULL",
        (args.uid,)).fetchone()
    if not row:
        return _err(f"no memory with uid {args.uid}")
    signal = 1 if args.signal == "up" else -1
    conn.execute(
        "INSERT INTO memory_feedback(memory_id, signal, note) VALUES (?,?,?)",
        (row["id"], signal, args.note))
    conn.commit()
    net = conn.execute(
        "SELECT COALESCE(SUM(signal),0) FROM memory_feedback WHERE memory_id = ?",
        (row["id"],)).fetchone()[0]
    print(f"{'👍' if signal > 0 else '👎'} {args.uid} — {row['title'][:60]} (net {net:+d})")
    return 0


def _pinned_blocks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    try:
        return list(conn.execute(
            "SELECT label, value, char_limit FROM blocks ORDER BY position, label"))
    except sqlite3.OperationalError:
        return []


def cmd_block(args) -> int:
    """Manage pinned core blocks (always-injected context)."""
    conn = connect()
    ensure_blocks_table(conn)
    action = args.action
    if action == "set":
        if not args.value:
            return _err("value required: brain block set <label> \"<value>\"")
        val = args.value
        if len(val) > args.char_limit:
            _warn(f"value {len(val)} chars > limit {args.char_limit}; truncating")
            val = val[:args.char_limit]
        conn.execute("""
            INSERT INTO blocks(label, value, char_limit, position, updated_at)
            VALUES (?,?,?,?, datetime('now'))
            ON CONFLICT(label) DO UPDATE SET
                value=excluded.value, char_limit=excluded.char_limit,
                position=excluded.position, updated_at=datetime('now')
        """, (args.label, val, args.char_limit, args.position))
        conn.commit()
        print(f"pinned [{args.label}] ({len(val)}/{args.char_limit} chars)")
    elif action == "rm":
        cur = conn.execute("DELETE FROM blocks WHERE label = ?", (args.label,))
        conn.commit()
        print(f"removed [{args.label}]" if cur.rowcount else f"no block [{args.label}]")
    else:  # list
        rows = _pinned_blocks(conn)
        if not rows:
            print("(no pinned blocks — brain block set <label> \"<value>\")")
            return 0
        total = sum(len(r["value"]) for r in rows)
        for r in rows:
            print(f"[{r['label']}] ({len(r['value'])}/{r['char_limit']}) {r['value']}")
        print(f"\n{len(rows)} block(s), {total} chars (~{total // 4} tokens) injected per context")
    return 0


def cmd_get(args) -> int:
    conn = connect()
    r = conn.execute("""
        SELECT m.*, GROUP_CONCAT(t.name, ',') AS all_tags
        FROM memories m
        LEFT JOIN memory_tags mt ON mt.memory_id = m.id
        LEFT JOIN tags t ON t.id = mt.tag_id
        WHERE m.uid = ? AND m.deleted_at IS NULL
        GROUP BY m.id
    """, (args.uid,)).fetchone()
    if not r:
        return _err(f"uid {args.uid} not found")

    # Update access tracking + recall log (re-extraction guard input).
    conn.execute("UPDATE memories SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?", (r["id"],))
    conn.commit()
    _log_recall(conn, [r["id"]])

    keys = r.keys()
    invalid_at = r["invalid_at"] if "invalid_at" in keys else None
    superseded = r["superseded_by"] if "superseded_by" in keys else None
    if args.json:
        d = dict(r)
        d.pop("content_hash", None)
        print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"\n{r['title']}")
        print(f"[{r['canonical_type'] or r['type']}] {r['uid']} · {r['created_at']}")
        if invalid_at:
            note = f" — superseded by {superseded} (`brain get {superseded}`)" if superseded else ""
            print(f"⚠ INVALIDATED {invalid_at}{note}")
        if r["project"]:
            print(f"project: {r['project']}")
        if r["all_tags"]:
            print(f"tags: {r['all_tags']}")
        print(f"\n{r['content'] or ''}\n")
        # Artifacts: the pointer is only useful with its CURRENT state — a
        # memory that cites a file is worth much less if the file is gone or
        # has moved on since the fact was written.
        arts = _artifacts_for(conn, r["id"])
        if arts:
            print("artifacts:")
            for a in arts:
                if a["missing_at"]:
                    state = f"MISSING since {str(a['missing_at'])[:10]}"
                elif a["changed_at"]:
                    state = f"CHANGED since {str(a['changed_at'])[:10]} · {_fmt_size(a['size'])}"
                else:
                    state = f"{_fmt_size(a['size'])} · {str(a['mtime'] or '')[:10]}"
                print(f"  {a['path']}  [{state}]")
            print()
    return 0


def cmd_link(args) -> int:
    conn = connect()
    if args.kind not in LINK_KINDS:
        return _err(f"invalid kind '{args.kind}'. Valid: {sorted(LINK_KINDS)}")
    src = conn.execute("SELECT id FROM memories WHERE uid = ?", (args.src,)).fetchone()
    dst = conn.execute("SELECT id FROM memories WHERE uid = ?", (args.dst,)).fetchone()
    if not src or not dst:
        return _err(f"src or dst uid not found ({args.src} → {args.dst})")
    try:
        conn.execute("INSERT INTO memory_links(src_id, dst_id, kind) VALUES (?, ?, ?)",
                     (src["id"], dst["id"], args.kind))
        conn.commit()
        print(f"linked {args.src} --[{args.kind}]--> {args.dst}")
        return 0
    except sqlite3.IntegrityError:
        return _err("link already exists")


def cmd_delete(args) -> int:
    conn = connect()
    cursor = conn.execute(
        "UPDATE memories SET deleted_at = datetime('now') WHERE uid = ? AND deleted_at IS NULL",
        (args.uid,),
    )
    if cursor.rowcount == 0:
        conn.commit()
        return _err(f"uid {args.uid} not found or already deleted")
    log_alteration(conn, args.uid, "delete", delta=None, reason=None)
    conn.commit()
    print(f"deleted {args.uid}")
    return 0


def cmd_restore(args) -> int:
    conn = connect()
    cursor = conn.execute(
        "UPDATE memories SET deleted_at = NULL WHERE uid = ? AND deleted_at IS NOT NULL",
        (args.uid,),
    )
    conn.commit()
    if cursor.rowcount == 0:
        return _err(f"uid {args.uid} not found or not deleted")
    print(f"restored {args.uid}")
    return 0


def cmd_update(args) -> int:
    """Mutate an existing memory's content (append or replace), log an alteration.

    --append "<text>"   → append a dated bullet to content
    --replace "<text>"  → replace the full content

    Keeps content_hash, updated_at, FTS (trigger-synced), and embeddings in sync.
    """
    if not args.append and args.replace is None and not getattr(args, "abstract", None):
        return _err("provide --append \"<text>\", --replace \"<full content>\", or --abstract")
    if args.append and args.replace is not None:
        return _err("use only one of --append / --replace, not both")

    conn = connect(load_vec=True)
    cursor = conn.cursor()

    row = cursor.execute(
        "SELECT id, title, content, deleted_at FROM memories WHERE uid = ?",
        (args.uid,),
    ).fetchone()
    if not row:
        return _err(f"uid {args.uid} not found")
    if row["deleted_at"] is not None:
        return _err(f"uid {args.uid} is soft-deleted — `brain restore {args.uid}` first")

    memory_id = row["id"]
    title = row["title"]
    old_content = row["content"] or ""

    kind = None
    if args.append:
        text = args.append.strip()
        if not text:
            return _err("--append text must not be empty")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bullet = f"\n\n- [{today}] {text}"
        new_content = old_content + bullet
        kind = "append"
        delta = bullet.strip()
    elif args.replace is not None:
        new_content = args.replace
        kind = "replace"
        delta = new_content
    else:
        new_content = old_content  # --abstract only: content untouched

    if kind:
        new_hash = content_hash(title, new_content)
        # The AFTER UPDATE trigger on memories re-syncs memories_fts automatically,
        # and a BEFORE UPDATE trigger snapshots the old content into memory_versions.
        cursor.execute(
            "UPDATE memories SET content = ?, content_hash = ?, updated_at = datetime('now') "
            "WHERE id = ?",
            (new_content, new_hash, memory_id),
        )
        log_alteration(conn, args.uid, kind, delta=delta, reason=args.reason)

    if getattr(args, "abstract", None):
        if _schema_flags(conn)["abstract"]:
            cursor.execute("UPDATE memories SET abstract = ? WHERE id = ?",
                           (args.abstract.strip(), memory_id))
            if not kind:
                log_alteration(conn, args.uid, "abstract", delta=args.abstract.strip()[:80],
                               reason=args.reason)
        else:
            _warn("--abstract ignored — run `brain migrate` (schema v4)")

    # Re-embed in the SAME transaction, then one commit (see _embed_in_txn).
    if kind and _vec_loaded:
        _embed_in_txn(conn, cursor, memory_id, title, new_content, args.uid)

    conn.commit()

    print(args.uid)
    return 0


def cmd_invalidate(args) -> int:
    """Soft fact invalidation (Zep's bi-temporal lesson): wrong/superseded
    knowledge is marked invalid_at, never deleted. Default search excludes it;
    `--include-invalid` / `--as-of <date>` still see it. History stays intact.
    """
    conn = connect()
    if not _schema_flags(conn)["invalid_at"]:
        return _err("invalid_at column missing — run `brain migrate` (schema v4)")

    row = conn.execute(
        "SELECT id, invalid_at FROM memories WHERE uid = ? AND deleted_at IS NULL",
        (args.uid,),
    ).fetchone()
    if not row:
        return _err(f"uid {args.uid} not found")

    if args.undo:
        if row["invalid_at"] is None:
            return _err(f"{args.uid} is not invalidated")
        conn.execute("UPDATE memories SET invalid_at = NULL, superseded_by = NULL WHERE id = ?",
                     (row["id"],))
        log_alteration(conn, args.uid, "revalidate", reason=args.reason)
        conn.commit()
        print(f"revalidated {args.uid}")
        return 0

    if row["invalid_at"] is not None:
        return _err(f"{args.uid} already invalidated at {row['invalid_at']} (--undo to restore)")

    keeper = None
    if args.superseded_by:
        keeper = conn.execute(
            "SELECT id, uid FROM memories WHERE uid = ? AND deleted_at IS NULL",
            (args.superseded_by,),
        ).fetchone()
        if not keeper:
            return _err(f"--superseded-by uid {args.superseded_by} not found")

    conn.execute(
        "UPDATE memories SET invalid_at = datetime('now'), superseded_by = ? WHERE id = ?",
        (keeper["uid"] if keeper else None, row["id"]),
    )
    log_alteration(conn, args.uid, "invalidate",
                   delta=f"superseded_by {keeper['uid']}" if keeper else None,
                   reason=args.reason)
    if keeper:
        try:
            conn.execute("INSERT INTO memory_links(src_id, dst_id, kind) VALUES (?, ?, 'superseded_by')",
                         (row["id"], keeper["id"]))
        except sqlite3.IntegrityError:
            pass
    conn.commit()
    print(f"invalidated {args.uid}" + (f" → superseded by {keeper['uid']}" if keeper else ""))
    return 0


# Reconcile similarity thresholds (Mem0's AUDN loop, decided by the CALLING
# agent instead of an API call — brain itself stays LLM-free).
SIM_NOOP_RECALLED = 0.92   # ≥ this AND recently recalled → re-extraction echo
SIM_UPDATE = 0.85          # ≥ this → same topic, merge into the neighbor
SIM_REVIEW = 0.70          # ≥ this → ambiguous, agent must look


def _save_ns(**kw) -> argparse.Namespace:
    """Namespace with every field cmd_save reads, so programmatic callers
    (reconcile --auto, review) cannot AttributeError when save grows a flag."""
    base = dict(
        type="note", title="", content="", project="", area="", tags=[],
        source_file="", force=False, no_embed=False, abstract=None,
        identity=None, anchor=None,
        status=None, priority=None, energy=None, points=None,
        due_at=None, external_ref=None, parent_uid=None,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _reconcile_decide(conn, cursor, title: str, content: str, args) -> dict:
    """The AUDN decision core: candidate fact → suggestion packet.
    Shared by `brain reconcile` (agent-facing) and `brain harvest` (automatic)."""
    h = content_hash(title, content)
    flags = _schema_flags(conn)
    invalid_sql = " AND invalid_at IS NULL" if flags["invalid_at"] else ""
    exact = cursor.execute(
        f"SELECT uid FROM memories WHERE content_hash = ? AND deleted_at IS NULL{invalid_sql} LIMIT 1",
        (h,),
    ).fetchone()

    # Declared identity beats every similarity heuristic. If the caller said
    # "this fact IS the deploy-runbook for corena", there is exactly one such
    # memory by construction and a near-miss embedding cannot override that.
    ident = getattr(args, "identity", None)
    ikey = identity_key(getattr(args, "type", "note"), ident) if ident else ""
    ident_hit = None
    if ikey and flags.get("identity_key"):
        ident_hit = cursor.execute(
            f"SELECT uid FROM memories WHERE identity_key = ? AND deleted_at IS NULL"
            f"{invalid_sql} LIMIT 1", (ikey,)).fetchone()

    filter_sql, filter_params = _build_filters(conn, args)
    top, explain = _hybrid_search(
        conn, cursor, f"{title} {content[:300]}".strip(), 5, filter_sql,
        filter_params, no_semantic=getattr(args, "no_semantic", False), want_explain=True,
    )
    recalled = _recently_recalled(conn)
    neighbors = []
    for score, r in top:
        sim = (explain or {}).get(r["id"], {}).get("sim")
        neighbors.append({
            "uid": r["uid"], "type": r["type"], "title": r["title"],
            "project": r["project"], "sim": sim, "score": round(score, 3),
            "abstract": r["abstract"] or _auto_abstract(r["content"], 160),
            "recalled_recently": r["id"] in recalled,
        })

    best = neighbors[0] if neighbors else None
    best_sim = (best or {}).get("sim") or 0.0
    if exact:
        suggestion, reason, target = "noop", f"exact content_hash duplicate of {exact['uid']}", exact["uid"]
    elif ident_hit:
        suggestion, reason, target = "update", (
            f"identity '{ident}' is already held by {ident_hit['uid']} — "
            f"same fact by declaration, not by similarity; "
            f"`brain update {ident_hit['uid']} --content \"...\"`"), ident_hit["uid"]
    elif best and best_sim >= SIM_NOOP_RECALLED and best["recalled_recently"]:
        suggestion, reason, target = "noop", (
            f"{best['uid']} (sim {best_sim:.2f}) was recalled into context in the last 24h — "
            "this is almost certainly a re-extraction echo, not new knowledge"), best["uid"]
    elif best and best_sim >= SIM_UPDATE:
        suggestion, reason, target = "update", (
            f"{best['uid']} covers the same topic (sim {best_sim:.2f}) — "
            f"`brain update {best['uid']} --append \"...\"`"), best["uid"]
    elif best and (best_sim >= SIM_REVIEW or best["sim"] is None):
        suggestion, reason, target = "review", (
            "close neighbors exist" + ("" if best["sim"] is not None else
            " (semantic unavailable — FTS neighbors only)") +
            " — read them and pick update / save / invalidate"), None
    else:
        suggestion, reason, target = "add", "no sufficiently similar memory found", None

    return {
        "suggestion": suggestion, "reason": reason, "target_uid": target,
        "candidate": {"title": title, "content_hash": h, "type": getattr(args, "type", "note"),
                      "identity": ident, "identity_key": ikey or None},
        "neighbors": neighbors,
    }


def cmd_reconcile(args) -> int:
    """ADD/UPDATE/NOOP decision packet for a candidate fact (search-then-merge,
    enforced). The agent calls this INSTEAD of blind `brain save`:

      suggestion=add     → safe to save (use --auto to do it in one step)
      suggestion=noop    → exact duplicate, or a re-extraction echo of a
                           memory recalled in the last 24h (Mem0's 808-dup
                           feedback loop — the guard exists because of it)
      suggestion=update  → same topic exists: `brain update <uid> --append`
      suggestion=review  → ambiguous; agent reads neighbors and decides
                           (update / save / invalidate the old one)

    --auto: applies 'add' (saves, prints uid), exits 2 on 'noop' (like dedup),
    exits 3 on 'update'/'review' (packet printed — agent must decide).
    """
    conn = connect(load_vec=True)
    cursor = conn.cursor()
    title = args.title.strip()
    content = (args.content or "").strip()
    if not title:
        return _err("--title is required")

    packet = _reconcile_decide(conn, cursor, title, content, args)
    suggestion, target = packet["suggestion"], packet["target_uid"]

    if args.auto:
        if suggestion == "add":
            return cmd_save(_save_ns(
                type=args.type, title=title, content=content,
                project=args.project, tags=args.tags, no_embed=args.no_embed,
                abstract=getattr(args, "abstract", None),
                identity=getattr(args, "identity", None),
                anchor=getattr(args, "anchor", None),
            ))
        print(json.dumps(packet, ensure_ascii=False, indent=2), file=sys.stderr)
        if suggestion == "noop":
            print(target)
            return 2
        return 3  # update/review — the agent must decide

    print(json.dumps(packet, ensure_ascii=False, indent=2))
    return 0


# ── harvest (automatic extraction from agent transcripts) ───────────────────

HARVEST_MAX_DELTA = 60_000  # chars of conversation per extraction call (newest kept)

HARVEST_PROMPT = """You are a memory-extraction gate for a personal knowledge DB. Below is the
newest delta of a coding-agent conversation (project hint: {project}).

Extract ONLY facts with lasting value (useful to recall in 1+ months): root
causes found, decisions made and WHY, bug fixes (symptom + cause + fix),
durable configs/IDs/paths, hard-won gotchas, ownership facts about people.

REJECT — do not output: transient task state, plans not yet executed,
restated instructions, tool output noise, file contents, chit-chat,
secrets/credentials (never copy a secret value anywhere), and anything that
reads like ALREADY-RECALLED memory (e.g. lines under "Relevant memories
(brain)" or text citing brain uids) — those are echoes, not new knowledge.

ANCHOR every fact to the ONE concrete entity it is about — the repo, service,
client, person, or system it would be recalled alongside. A fact with no such
entity is usually too vague to be worth keeping; prefer rewriting it as a
lesson about a specific thing over emitting a floating generality.

Output a JSON array, NOTHING else. Each item:
{{"type": "learning|decision|bug|snippet|note",
  "title": "<concise, searchable, under 100 chars>",
  "content": "<the fact, self-contained, with its WHY>",
  "project": "<slug, default {project}>",
  "anchor": "<the single entity this is about: repo/service/client/person>",
  "tags": "<3-5,comma,separated>",
  "abstract": "<one informative sentence>"}}
Output [] if nothing qualifies. Maximum 5 items — pick the most durable.

CONVERSATION DELTA:
{conversation}"""


REVIEW_PATH = Path.home() / ".config/brain/harvest-review.jsonl"

# Auto-triage thresholds for `brain review --auto`. Deliberately conservative:
# the queue exists because these cases were AMBIGUOUS, so auto-resolution only
# touches the two ends where the packet is not actually ambiguous at all.
AUTO_DROP_SIM = 0.90    # ≥ this to a live neighbor → near-certain duplicate
AUTO_KEEP_SIM = 0.72    # ≤ this → nothing close; the "review" call was noise


# Durability gate for `brain review --judge`. Same standard as HARVEST_PROMPT's
# REJECT clause, applied retroactively: reconcile only ever measured
# DUPLICATION, so queued candidates were never re-checked for whether they are
# worth keeping at all. An audit of 40 of the 780 auto-savable candidates on
# 2026-07-27 found 27% transient junk (point-in-time status, phase tracking,
# lead counts) — and best-neighbor similarity did NOT separate it from the good
# ones (0.546 vs 0.569), so no threshold tuning can substitute for this pass.
JUDGE_PROMPT = """You are auditing candidate memories before they are written into a personal
long-term knowledge DB used by coding agents. The bar: will this still be
USEFUL TO RECALL IN 1+ MONTHS?

USABLE = durable knowledge: root causes, decisions and WHY, bug
symptom/cause/fix, configs/IDs/paths that persist, hard-won gotchas,
ownership facts, verified conclusions.

JUNK = transient task state, plans not yet executed, restated instructions,
tool-output noise, ephemeral status ("now running", "phase 3 shipped"),
point-in-time counts/metrics that decay, vague generalities with no specific
subject, specs derivable from code or tickets.

Be strict — this DB already has a junk problem.

Output ONLY a JSON array, one object per item, no prose and no code fences:
{{"i": <index>, "verdict": "usable"|"junk"}}

ITEMS:
{items}"""

JUDGE_BATCH = 40


def _judge_durability(cands: list[dict], model: str) -> set[int]:
    """Indices judged JUNK. Fails SAFE: a failed/unparseable batch returns no
    junk for that batch, so a model hiccup can never silently delete facts."""
    import subprocess
    import tempfile

    junk: set[int] = set()
    for start in range(0, len(cands), JUDGE_BATCH):
        batch = cands[start:start + JUDGE_BATCH]
        payload = json.dumps([
            {"i": start + n, "type": c.get("type"), "title": c.get("title"),
             "content": (c.get("content") or "")[:700], "project": c.get("project")}
            for n, c in enumerate(batch)
        ], ensure_ascii=False, indent=1)
        env = {**os.environ, "BRAIN_HARVEST": "1"}
        try:
            r = subprocess.run(
                ["claude", "-p", JUDGE_PROMPT.format(items=payload), "--model", model],
                capture_output=True, text=True, timeout=300,
                cwd=tempfile.gettempdir(), env=env,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            _warn(f"judge batch @{start} failed ({e}) — keeping all of it")
            continue
        if r.returncode != 0:
            _warn(f"judge batch @{start} exited {r.returncode} — keeping all of it")
            continue
        m = re.search(r"\[.*\]", r.stdout, re.S)
        if not m:
            _warn(f"judge batch @{start} returned no JSON — keeping all of it")
            continue
        try:
            verdicts = json.loads(m.group(0))
        except ValueError:
            _warn(f"judge batch @{start} returned bad JSON — keeping all of it")
            continue
        seen = 0
        for v in verdicts:
            if not isinstance(v, dict):
                continue
            i = v.get("i")
            if isinstance(i, int) and start <= i < start + len(batch):
                seen += 1
                if str(v.get("verdict", "")).lower() == "junk":
                    junk.add(i)
        print(f"  judged {start + len(batch)}/{len(cands)} "
              f"({len(junk)} junk so far)", file=sys.stderr)
    return junk


def _review_load(path: Path) -> list[dict]:
    """Read the queue. Skips malformed lines rather than dying — this file is
    appended to by a background hook and a single bad line must not block the
    whole backlog."""
    items, bad = [], 0
    if not path.exists():
        return []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            d["_line"] = i
            items.append(d)
        except json.JSONDecodeError:
            bad += 1
    if bad:
        _warn(f"{bad} malformed line(s) in {path} skipped")
    return items


def _review_rewrite(path: Path, keep: list[dict]) -> None:
    """Atomic rewrite (same pattern as the rest of the file's writes)."""
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for d in keep:
            d.pop("_line", None)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    tmp.replace(path)


def cmd_review(args) -> int:
    """Triage the harvest review queue — the ambiguous candidates `brain
    harvest` refused to decide alone.

    Without this the queue is write-only: 993 entries had accumulated by
    2026-07-26 with no way to drain them, which quietly made the
    reconcile-gated capture pipeline lossy.

      brain review                 summary + the next N packets
      brain review --auto          resolve only the unambiguous ends
      brain review --resolve <n> --action drop|save|update --uid <uid>
      brain review --clear         drop everything (asks for --yes)
    """
    conn = connect(load_vec=True)
    cursor = conn.cursor()
    path = Path(args.file) if args.file else REVIEW_PATH
    items = _review_load(path)
    if not items:
        print(f"review queue empty ({path})")
        return 0

    def best_sim(d: dict) -> float:
        ns = (d.get("packet") or {}).get("neighbors") or []
        sims = [n.get("sim") for n in ns if isinstance(n.get("sim"), (int, float))]
        return max(sims) if sims else 0.0

    if args.clear:
        if not args.yes:
            return _err(f"--clear drops all {len(items)} queued candidates; re-run with --yes")
        _review_rewrite(path, [])
        print(f"cleared {len(items)} queued candidate(s)")
        return 0

    if args.resolve is not None:
        idx = args.resolve
        if not (0 <= idx < len(items)):
            return _err(f"index {idx} out of range (0..{len(items) - 1})")
        d = items[idx]
        cand = d.get("candidate") or {}
        if args.action == "drop":
            pass
        elif args.action == "save":
            if cmd_save(_save_ns(
                type=cand.get("type", "note"), title=cand.get("title", ""),
                content=cand.get("content", ""), project=cand.get("project") or "",
                tags=cand.get("tags") or [], abstract=cand.get("abstract"),
            )) != 0:
                return _err("save failed; queue left untouched")
        elif args.action == "update":
            if not args.uid:
                return _err("--action update needs --uid <target>")
            if cmd_update(argparse.Namespace(
                uid=args.uid, append=cand.get("content", ""), replace=None,
                abstract=None, reason="merged from harvest review queue",
            )) != 0:
                return _err("update failed; queue left untouched")
        else:
            return _err("--action must be drop|save|update")
        _review_rewrite(path, [x for i, x in enumerate(items) if i != idx])
        print(f"resolved #{idx} ({args.action}) — {len(items) - 1} left in queue")
        return 0

    if args.auto:
        # Pass 1 — DUPLICATION (reconcile, no LLM). Splits the queue three ways.
        dropped = 0
        savable: list[tuple[dict, dict]] = []   # (queue entry, candidate)
        ambiguous: list[tuple[dict, dict]] = []
        for d in items:
            cand = d.get("candidate") or {}
            title = (cand.get("title") or "").strip()
            content = (cand.get("content") or "").strip()
            if not title:
                dropped += 1          # unusable entry
                continue
            # re-decide against the CURRENT db — most of this backlog was
            # queued weeks ago and the neighbourhood has moved since
            ns = argparse.Namespace(type=cand.get("type", "note"), project=None,
                                    tags=None, no_semantic=args.no_semantic,
                                    identity=None, since=None, until=None)
            packet = _reconcile_decide(conn, cursor, title, content, ns)
            sim = 0.0
            for n in packet["neighbors"]:
                if isinstance(n.get("sim"), (int, float)):
                    sim = max(sim, n["sim"])
            if packet["suggestion"] == "noop" or sim >= AUTO_DROP_SIM:
                dropped += 1
            elif packet["suggestion"] == "add" and sim <= AUTO_KEEP_SIM:
                savable.append((d, cand))
            else:
                ambiguous.append((d, cand))

        # Pass 2 — DURABILITY (one LLM pass). Orthogonal to pass 1: similarity
        # cannot tell a durable fact from a transient status line. Junk is
        # dropped from BOTH buckets; ambiguous survivors stay queued because
        # what makes them ambiguous is duplication, which this pass never saw.
        judged_junk_save: set[int] = set()
        judged_junk_amb: set[int] = set()
        if args.judge:
            print(f"judging {len(savable)} savable + {len(ambiguous)} ambiguous "
                  f"candidate(s) for durability ({args.model})…", file=sys.stderr)
            judged_junk_save = _judge_durability([c for _, c in savable], args.model)
            judged_junk_amb = _judge_durability([c for _, c in ambiguous], args.model)

        saved = 0
        keep: list[dict] = []
        for i, (d, cand) in enumerate(savable):
            if i in judged_junk_save:
                dropped += 1
                continue
            if args.dry_run:
                saved += 1
                continue
            if cmd_save(_save_ns(
                type=cand.get("type", "note"), title=cand.get("title", "").strip(),
                content=(cand.get("content") or "").strip(),
                project=cand.get("project") or "", abstract=cand.get("abstract"),
                tags=cand.get("tags") or [], anchor=cand.get("anchor"),
            )) == 0:
                saved += 1
            else:
                keep.append(d)        # save failed → leave it queued, retry later
        for i, (d, _cand) in enumerate(ambiguous):
            if i in judged_junk_amb:
                dropped += 1
                continue
            keep.append(d)

        judged_note = (f", {len(judged_junk_save) + len(judged_junk_amb)} killed by the "
                       f"durability judge" if args.judge else "")
        if args.dry_run:
            print(f"DRY RUN: would drop {dropped}{judged_note}, save {saved}, "
                  f"leave {len(keep)} for manual review (of {len(items)})")
            return 0
        _review_rewrite(path, keep)
        print(f"auto-triage: dropped {dropped}{judged_note}, saved {saved} durable+new, "
              f"{len(keep)} still need a human (of {len(items)})")
        return 0

    # default: summarise + show the next N packets
    buckets = {"near-dup (≥0.90)": 0, "ambiguous": 0, "looks new (≤0.72)": 0}
    for d in items:
        s = best_sim(d)
        buckets["near-dup (≥0.90)" if s >= AUTO_DROP_SIM else
                "looks new (≤0.72)" if s <= AUTO_KEEP_SIM else "ambiguous"] += 1
    print(f"review queue: {len(items)} candidate(s) — {path}")
    for k, v in buckets.items():
        print(f"  {k:<20} {v}")
    print(f"\n  brain review --auto            resolve the {buckets['near-dup (≥0.90)'] + buckets['looks new (≤0.72)']} unambiguous ones")
    print("  brain review --auto --dry-run  preview that first")
    print()

    if args.json:
        print(json.dumps(items[:args.limit], ensure_ascii=False, indent=2))
        return 0
    for i, d in enumerate(items[:args.limit]):
        cand = d.get("candidate") or {}
        print(f"[{i}] ({cand.get('type', '?')}) {cand.get('title', '')[:78]}")
        print(f"     best neighbor sim {best_sim(d):.2f} · project={cand.get('project')}")
        for n in ((d.get("packet") or {}).get("neighbors") or [])[:2]:
            sim = n.get("sim")
            print(f"       ~ {n.get('uid')} {sim if sim is None else f'{sim:.2f}'} {str(n.get('title'))[:56]}")
        print(f"     resolve: brain review --resolve {i} --action drop|save|update --uid <uid>")
    return 0


ANCHOR_PROMPT = """For each memory below, name the ONE concrete entity it is about — the repo,
service, client, person, product, or system someone would recall it alongside.

Rules:
- A short noun phrase, lowercase, 1-3 words. Prefer the proper name when there
  is one ("corena", "falkor", "brain.db", "unify", "spright").
- Be CONSISTENT: the same entity must get byte-identical text every time.
- If the memory is genuinely about no single entity, use null. Do not invent
  a vague bucket like "general" or "misc" — a wrong anchor is worse than none,
  because it will pull unrelated memories into the same cluster.

Output ONLY a JSON array, no prose and no code fences:
{{"i": <index>, "anchor": "<entity>"|null}}

MEMORIES:
{items}"""


def cmd_anchor(args) -> int:
    """Backfill `anchor` on memories that lack one.

    Anchors only ever populated from new harvests, so the existing corpus had
    ~1% coverage and the feature could not affect anything. This backfills it
    in batches, then re-embeds the touched rows so the anchor actually reaches
    the index (FTS picks it up automatically via the v6 triggers).
    """
    conn = connect(load_vec=True)
    cursor = conn.cursor()
    if not _schema_flags(conn).get("anchor"):
        return _err("anchor column missing — run `brain migrate` (schema v5+)")

    rows = cursor.execute(
        "SELECT id, uid, canonical_type, title, content, project FROM memories "
        "WHERE (anchor IS NULL OR anchor = '') AND deleted_at IS NULL "
        "AND invalid_at IS NULL ORDER BY id DESC" + (f" LIMIT {int(args.limit)}" if args.limit else "")
    ).fetchall()
    if not rows:
        print("every live memory already has an anchor")
        return 0
    print(f"backfilling anchors for {len(rows)} memories ({args.model})…", file=sys.stderr)

    import subprocess
    import tempfile

    set_count, null_count, failed = 0, 0, 0
    for start in range(0, len(rows), JUDGE_BATCH):
        batch = rows[start:start + JUDGE_BATCH]
        payload = json.dumps([
            {"i": start + n, "type": r["canonical_type"], "title": r["title"],
             "content": (r["content"] or "")[:400], "project": r["project"]}
            for n, r in enumerate(batch)
        ], ensure_ascii=False, indent=1)
        env = {**os.environ, "BRAIN_HARVEST": "1"}
        try:
            res = subprocess.run(
                ["claude", "-p", ANCHOR_PROMPT.format(items=payload), "--model", args.model],
                capture_output=True, text=True, timeout=300,
                cwd=tempfile.gettempdir(), env=env)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            _warn(f"anchor batch @{start} failed ({e}) — left unanchored")
            failed += len(batch)
            continue
        m = re.search(r"\[.*\]", res.stdout, re.S) if res.returncode == 0 else None
        if not m:
            _warn(f"anchor batch @{start} unusable — left unanchored")
            failed += len(batch)
            continue
        try:
            out = json.loads(m.group(0))
        except ValueError:
            _warn(f"anchor batch @{start} bad JSON — left unanchored")
            failed += len(batch)
            continue

        for v in out:
            if not isinstance(v, dict):
                continue
            i = v.get("i")
            if not (isinstance(i, int) and start <= i < start + len(batch)):
                continue          # ignore indices the model invented
            anchor = v.get("anchor")
            anchor = " ".join(str(anchor).split()).lower()[:60] if anchor else ""
            if not anchor or anchor in {"general", "misc", "none", "null", "n/a"}:
                null_count += 1
                continue
            r = rows[i]
            if not args.dry_run:
                cursor.execute("UPDATE memories SET anchor = ? WHERE id = ?", (anchor, r["id"]))
                # re-embed so the anchor reaches the vector side too; FTS is
                # updated by the v6 trigger on this UPDATE automatically.
                if not args.no_embed and _vec_loaded:
                    embed_memory(conn, r["id"], r["title"] or "", r["content"] or "", anchor)
            set_count += 1
        if not args.dry_run:
            conn.commit()
        print(f"  {min(start + JUDGE_BATCH, len(rows))}/{len(rows)} "
              f"({set_count} anchored)", file=sys.stderr)

    verb = "would set" if args.dry_run else "set"
    print(f"{verb} {set_count} anchor(s); {null_count} had no single entity; "
          f"{failed} left unanchored (batch failure)")
    return 0


# ────────────────────────────────────────────────────────────────────────────
# v7: artifact graph — pointers to real files, with drift detection
# ────────────────────────────────────────────────────────────────────────────

# Absolute-ish paths worth tracking. Anchored at real roots so prose like
# "and/or" or "TODO/done" never registers as a file.
ARTIFACT_PATH_RE = re.compile(
    r"(?:~|/Users/[\w.\-]+|/opt|/etc|/var|/srv|/tmp)(?:/[\w.\-@+]+){1,12}/?")

# Above this, record size+mtime but skip the content hash. Hashing a 126MB db
# on every `artifact check` would make the command something you stop running,
# and size+mtime already catches the drift that matters.
ARTIFACT_HASH_CAP = 64 * 1024 * 1024


def ensure_artifact_tables(conn: sqlite3.Connection) -> None:
    """Idempotent (lazy mirror of the v7 migration)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS artifacts (
            id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, real_path TEXT NOT NULL,
            kind TEXT, size INTEGER, sha256 TEXT, mtime TEXT,
            first_seen TEXT NOT NULL DEFAULT (datetime('now')),
            last_checked TEXT, missing_at TEXT, changed_at TEXT
        )""")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_artifacts (
            memory_id INTEGER NOT NULL, artifact_id INTEGER NOT NULL,
            PRIMARY KEY (memory_id, artifact_id)
        )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_memart_artifact "
                 "ON memory_artifacts(artifact_id)")


def _artifact_state(real: Path) -> dict:
    """stat + (bounded) content hash. Never raises on a vanished/unreadable path."""
    try:
        st = real.stat()
    except (OSError, ValueError):
        return {"exists": False, "kind": None, "size": None, "sha256": None, "mtime": None}
    if real.is_dir():
        return {"exists": True, "kind": "dir", "size": None, "sha256": None,
                "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(" ", "seconds")}
    sha = None
    if st.st_size <= ARTIFACT_HASH_CAP:
        try:
            h = hashlib.sha256()
            with open(real, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    h.update(block)
            sha = h.hexdigest()[:32]
        except OSError:
            sha = None
    return {"exists": True, "kind": "file", "size": st.st_size, "sha256": sha,
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(" ", "seconds")}


def _artifact_upsert(conn: sqlite3.Connection, raw_path: str) -> int | None:
    """Register a path (idempotent), recording its CURRENT state. Returns id."""
    raw = raw_path.rstrip(".,)`'\"")
    if len(raw) < 8:
        return None
    real = Path(os.path.expanduser(raw))
    st = _artifact_state(real)
    row = conn.execute("SELECT id FROM artifacts WHERE path = ?", (raw,)).fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO artifacts(path, real_path, kind, size, sha256, mtime, last_checked, "
        "missing_at) VALUES (?,?,?,?,?,?, datetime('now'), ?)",
        (raw, str(real), st["kind"], st["size"], st["sha256"], st["mtime"],
         None if st["exists"] else _now()))
    return cur.lastrowid


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _artifacts_for(conn: sqlite3.Connection, memory_id: int) -> list[sqlite3.Row]:
    try:
        return list(conn.execute(
            "SELECT a.* FROM artifacts a JOIN memory_artifacts ma ON ma.artifact_id = a.id "
            "WHERE ma.memory_id = ? ORDER BY a.path", (memory_id,)))
    except sqlite3.OperationalError:
        return []


def _fmt_size(n: int | None) -> str:
    if n is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return str(n)


def cmd_artifact(args) -> int:
    """Track the real files memories point at, and whether they still exist.

    A memory cannot carry a 126MB dump, so it carries a pointer plus the
    file's state when the fact was written. `check` re-stats everything, so a
    memory can tell you not only where the file is but whether it still says
    what it said.
    """
    conn = connect()
    ensure_artifact_tables(conn)
    conn.commit()
    action = args.action

    if action == "scan":
        live = "deleted_at IS NULL AND invalid_at IS NULL"
        found, linked = 0, 0
        for r in conn.execute(f"SELECT id, title, content FROM memories WHERE {live}"):
            seen = set()
            for m in ARTIFACT_PATH_RE.finditer(f"{r['title']}\n{r['content'] or ''}"):
                raw = m.group(0).rstrip(".,)`'\"")
                if raw in seen:
                    continue
                seen.add(raw)
                if args.dry_run:
                    found += 1
                    continue
                aid = _artifact_upsert(conn, raw)
                if aid is None:
                    continue
                found += 1
                try:
                    conn.execute("INSERT INTO memory_artifacts(memory_id, artifact_id) "
                                 "VALUES (?,?)", (r["id"], aid))
                    linked += 1
                except sqlite3.IntegrityError:
                    pass
        if args.dry_run:
            print(f"DRY RUN: would register {found} path reference(s)")
            return 0
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        missing = conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE missing_at IS NOT NULL").fetchone()[0]
        print(f"scanned: {found} reference(s), {linked} new link(s)")
        print(f"artifacts: {total} tracked, {missing} missing ({missing * 100 // max(total, 1)}%)")
        return 0

    if action == "check":
        rows = list(conn.execute("SELECT * FROM artifacts"))
        if not rows:
            return _err("no artifacts tracked — run `brain artifact scan` first")
        vanished, restored, changed, ok = [], [], [], 0
        for a in rows:
            st = _artifact_state(Path(a["real_path"]))
            if not st["exists"]:
                if a["missing_at"] is None:
                    vanished.append(a)
                    conn.execute("UPDATE artifacts SET missing_at=?, last_checked=datetime('now') "
                                 "WHERE id=?", (_now(), a["id"]))
                else:
                    conn.execute("UPDATE artifacts SET last_checked=datetime('now') WHERE id=?",
                                 (a["id"],))
                continue
            drifted = (a["sha256"] and st["sha256"] and a["sha256"] != st["sha256"]) or \
                      (a["sha256"] is None and a["size"] is not None and st["size"] != a["size"])
            if a["missing_at"] is not None:
                restored.append(a)
            elif drifted:
                changed.append(a)
            else:
                ok += 1
            conn.execute(
                "UPDATE artifacts SET kind=?, size=?, sha256=?, mtime=?, missing_at=NULL, "
                "last_checked=datetime('now'), changed_at=CASE WHEN ? THEN ? ELSE changed_at END "
                "WHERE id=?",
                (st["kind"], st["size"], st["sha256"], st["mtime"],
                 1 if drifted else 0, _now(), a["id"]))
        conn.commit()
        print(f"checked {len(rows)} artifact(s): {ok} unchanged, {len(changed)} changed, "
              f"{len(vanished)} newly missing, {len(restored)} reappeared")
        for label, group in (("CHANGED since recorded", changed),
                             ("NEWLY MISSING", vanished), ("REAPPEARED", restored)):
            if not group:
                continue
            print(f"\n{label}:")
            for a in group[:args.limit]:
                n = conn.execute("SELECT COUNT(*) FROM memory_artifacts WHERE artifact_id=?",
                                 (a["id"],)).fetchone()[0]
                print(f"  {a['path']}  ({n} memories, was {_fmt_size(a['size'])})")
            if len(group) > args.limit:
                print(f"  … {len(group) - args.limit} more")
        return 0

    # list
    where = ""
    if args.missing:
        where = "WHERE missing_at IS NOT NULL"
    elif args.large:
        where = "WHERE size >= 200000 AND missing_at IS NULL"
    rows = list(conn.execute(
        f"SELECT a.*, (SELECT COUNT(*) FROM memory_artifacts m WHERE m.artifact_id=a.id) n "
        f"FROM artifacts a {where} ORDER BY n DESC, a.size DESC LIMIT ?", (args.limit,)))
    if not rows:
        print("(no artifacts — `brain artifact scan`)")
        return 0
    for a in rows:
        flag = "GONE" if a["missing_at"] else ("DRIFT" if a["changed_at"] else "ok  ")
        print(f"  {flag}  {_fmt_size(a['size']):>8}  {a['n']:3} mem  {a['path']}")
    return 0


# Secret patterns → env-var prefix. Deliberately narrow: high-confidence,
# structurally distinctive tokens only. A loose pattern here would rewrite
# innocent content, and the whole operation is a content mutation.
# Two failure modes to avoid, both hit on the first live run (2026-07-27):
#
# TOO NARROW — `secret_[A-Za-z0-9]{20,}` stopped at the first '-' or '_', so a
# real token like `secret_plB3nYUn-_oK7…` was matched only in part or not at
# all, and the follow-up scan then reported "no secrets" with false
# confidence. Credential alphabets include - _ + / = ; the body classes below
# all do too.
#
# TOO BROAD — `A[A-Za-z0-9_\-]{40,}` matches ANY 41-char token beginning with
# "A", including opaque resource IDs sitting in URLs. An unanchored high-
# entropy pattern will eat real content. Anything that is not
# self-identifying by prefix now REQUIRES a context cue (`Bearer `).
SECRET_PATTERNS: list[tuple[str, str, str]] = [
    ("aws_access_key_id", r"AKIA[0-9A-Z]{16}", "AWS_ACCESS_KEY_ID"),
    ("github_pat", r"gh[pousr]_[A-Za-z0-9_\-]{20,}", "GITHUB_TOKEN"),
    ("openai_key", r"sk-[A-Za-z0-9_\-]{32,}", "OPENAI_API_KEY"),
    ("slack_token", r"xox[baprs]-[A-Za-z0-9-]{10,}", "SLACK_TOKEN"),
    ("scout_key", r"sk_[A-Za-z0-9+/=_\-]{20,}", "SCOUT_API_KEY"),
    ("scout_secret", r"secret_[A-Za-z0-9+/=_\-]{20,}", "SCOUT_ORG_SECRET"),
    # context-anchored: only a token presented AS a bearer credential
    ("bearer_token", r"(?<=Bearer )[A-Za-z0-9+/=_\-]{30,}", "BEARER_TOKEN"),
]

# Obvious non-secrets that structurally match. A placeholder rewritten into a
# ${VAR} would destroy the instruction the memory exists to convey.
SECRET_PLACEHOLDER = re.compile(
    r"^(\$|<|\{)|[<>]|^(YOUR|REDACTED|EXAMPLE|xxx+|\.\.\.)", re.I)

SECRETS_ENV_DEFAULT = Path.home() / ".config/brain/secrets.env"


def _scan_secrets(conn: sqlite3.Connection) -> list[dict]:
    """Every secret-looking token in live memory content.

    Returns records with the VALUE included — callers must never print it.
    """
    out = []
    for r in conn.execute(
        "SELECT id, uid, title, content FROM memories "
        "WHERE deleted_at IS NULL AND invalid_at IS NULL AND content IS NOT NULL"
    ):
        content = r["content"] or ""
        for kind, pat, prefix in SECRET_PATTERNS:
            for m in re.finditer(pat, content):
                val = m.group(0)
                # already replaced by a previous run, or a placeholder the
                # memory is deliberately telling you to substitute
                if val.startswith("${") or SECRET_PLACEHOLDER.search(val):
                    continue
                # never rewrite something already inside a ${...} reference
                if content[max(0, m.start() - 2):m.start()] == "${":
                    continue
                out.append({"id": r["id"], "uid": r["uid"], "title": r["title"],
                            "kind": kind, "prefix": prefix, "value": val})
    return out


def _env_load(path: Path) -> dict[str, str]:
    """Existing VAR=value pairs, so re-runs reuse names instead of duplicating."""
    got: dict[str, str] = {}
    if not path.exists():
        return got
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        got[k.strip()] = v.strip().strip("'\"")
    return got


def cmd_secrets(args) -> int:
    """Move secret values out of memory content into a chmod-600 env file,
    leaving a ${VAR} reference behind.

    Memories get injected into agent context by `search`/`context`, so a live
    credential stored in one can end up echoed into a transcript. Deleting the
    memory would lose the runbook it lives in; this keeps the note useful and
    moves only the value.

    Never prints a secret value — only counts, kinds and var names.
    """
    conn = connect(load_vec=True)
    env_path = Path(args.env_file) if args.env_file else SECRETS_ENV_DEFAULT
    if not args.extract:
        args.scan = True
    found = _scan_secrets(conn)
    if not found:
        print("no secret-looking values in live memory content")
        return 0

    existing = _env_load(env_path)
    val_to_var = {v: k for k, v in existing.items()}
    counters: dict[str, int] = {}
    for var in existing:
        base = var.rsplit("_", 1)[0]
        try:
            counters[base] = max(counters.get(base, 0), int(var.rsplit("_", 1)[1]))
        except (ValueError, IndexError):
            pass

    by_kind: dict[str, int] = {}
    for f in found:
        by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
    uniq = {f["value"] for f in found}
    print(f"{len(found)} occurrence(s) of {len(uniq)} distinct value(s) "
          f"across {len({f['uid'] for f in found})} memories")
    for k, n in sorted(by_kind.items()):
        print(f"  {k:20} {n}")

    if args.scan:
        print(f"\nvalues NOT shown. move them out with: brain secrets --extract "
              f"(env file: {env_path})")
        return 0

    # assign a var per DISTINCT value so the same credential is one variable
    assigned: dict[str, str] = {}
    for f in found:
        if f["value"] in val_to_var:
            assigned[f["value"]] = val_to_var[f["value"]]
            continue
        if f["value"] in assigned:
            continue
        counters[f["prefix"]] = counters.get(f["prefix"], 0) + 1
        assigned[f["value"]] = f"{f['prefix']}_{counters[f['prefix']]}"

    if args.dry_run:
        print(f"\nDRY RUN — would write {len(set(assigned.values()) - set(existing))} "
              f"new var(s) to {env_path} and rewrite {len(found)} occurrence(s):")
        for f in found:
            print(f"  {f['uid']}  {f['kind']:18} → ${{{assigned[f['value']]}}}  "
                  f"({f['title'][:44]})")
        return 0

    # 1. env file first — never rewrite content before the value is saved
    env_path.parent.mkdir(parents=True, exist_ok=True)
    new_lines = []
    for val, var in assigned.items():
        if var not in existing:
            new_lines.append(f"{var}='{val}'")
    if new_lines:
        header = "" if env_path.exists() else (
            "# Secrets extracted from brain.db by `brain secrets --extract`.\n"
            "# Memories reference these as ${VAR}. chmod 600 — never commit.\n")
        with open(env_path, "a") as f:
            if header:
                f.write(header)
            f.write("\n".join(new_lines) + "\n")
    os.chmod(env_path, 0o600)

    # 2. rewrite content, refresh hash + embedding, log the alteration
    touched = 0
    for f in found:
        row = conn.execute("SELECT title, content, anchor FROM memories WHERE id = ?",
                           (f["id"],)).fetchone() if _schema_flags(conn).get("anchor") else \
            conn.execute("SELECT title, content, NULL AS anchor FROM memories WHERE id = ?",
                         (f["id"],)).fetchone()
        if not row or f["value"] not in (row["content"] or ""):
            continue          # already replaced by an earlier occurrence
        new_content = (row["content"] or "").replace(
            f["value"], "${" + assigned[f["value"]] + "}")
        if f"see {env_path}" not in new_content:
            new_content += f"\n\n[secret values moved to {env_path} — reference by ${{VAR}}]"
        conn.execute("UPDATE memories SET content = ?, content_hash = ? WHERE id = ?",
                     (new_content, content_hash(row["title"] or "", new_content), f["id"]))
        log_alteration(conn, f["uid"], "redact",
                       delta=f"{f['kind']} → ${{{assigned[f['value']]}}}",
                       reason="secret moved to env file")
        if not args.no_embed and _vec_loaded:
            embed_memory(conn, f["id"], row["title"] or "", new_content, _row_get(row, "anchor"))
        touched += 1
    conn.commit()
    print(f"\nmoved {len(set(assigned.values()))} distinct value(s) → {env_path} (chmod 600)")
    print(f"rewrote {touched} memory occurrence(s) to ${{VAR}} references")
    print("load them with:  set -a && . " + str(env_path) + " && set +a")
    return 0


def ensure_harvest_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS harvest_state (
            path         TEXT PRIMARY KEY,
            byte_offset  INTEGER NOT NULL DEFAULT 0,
            last_run     DATETIME
        )
    """)


def _transcript_delta(path: Path, offset: int) -> tuple[str, int]:
    """Clean USER/ASSISTANT text from JSONL bytes past `offset`.

    Only consumes up to the last complete line — a transcript mid-write never
    corrupts the watermark. Tool calls/results are dropped: facts live in the
    prose, and tool dumps are exactly the noise Mem0's junk audit drowned in.
    """
    size = path.stat().st_size
    if size <= offset:
        return "", offset
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read()
    cut = raw.rfind(b"\n")
    if cut < 0:
        return "", offset  # no complete new line yet
    raw = raw[:cut + 1]
    new_offset = offset + len(raw)

    out = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") not in ("user", "assistant"):
            continue
        content = (d.get("message") or {}).get("content")
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(c.get("text", "") for c in content
                         if isinstance(c, dict) and c.get("type") == "text")
        text = "\n".join(t for t in texts if t).strip()
        if not text or text.startswith("<system-reminder"):
            continue
        out.append(f"{'USER' if d['type'] == 'user' else 'ASSISTANT'}: {text}")
    return "\n\n".join(out), new_offset


def _extract_candidates(prompt: str, model: str) -> list[dict] | None:
    """One headless `claude -p` call → candidate facts. None = call failed
    (caller must NOT advance the watermark); [] = nothing worth saving."""
    import subprocess
    import tempfile
    env = {**os.environ, "BRAIN_HARVEST": "1"}  # its Stop hook must not re-harvest
    try:
        r = subprocess.run(
            ["claude", "-p", prompt, "--model", model],
            capture_output=True, text=True, timeout=240,
            cwd=tempfile.gettempdir(),  # neutral cwd: no project CLAUDE.md pulled in
            env=env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        _warn(f"claude extraction failed: {e}")
        return None
    if r.returncode != 0:
        _warn(f"claude -p exited {r.returncode}: {(r.stderr or '')[:200]}")
        return None
    m = re.search(r"\[.*\]", r.stdout, re.S)
    if not m:
        return []
    try:
        out = json.loads(m.group(0))
    except ValueError:
        _warn("extraction output was not valid JSON; will retry next harvest")
        return None
    return [c for c in out if isinstance(c, dict)] if isinstance(out, list) else []


def cmd_harvest(args) -> int:
    """Automatic extraction: agent transcript → candidate facts → reconcile.

    Closes the discipline gap (Mem0-style zero-cooperation capture) without
    Mem0's failure modes: one LLM call per session-delta (not per message),
    a REJECT gate in the prompt, and EVERY candidate goes through the
    reconcile pipeline — exact dups and recall-echoes die before storage.
    Watermark (harvest_state) means bytes are never reprocessed.
    """
    if os.environ.get("BRAIN_HARVEST") == "1":
        return 0  # spawned from inside an extraction call — never recurse

    path = Path(args.transcript).expanduser()
    if not path.exists():
        return _err(f"transcript not found: {path}")

    conn = connect(load_vec=True)
    cursor = conn.cursor()
    ensure_harvest_table(conn)
    row = cursor.execute(
        "SELECT byte_offset FROM harvest_state WHERE path = ?", (str(path),)
    ).fetchone()
    offset = row["byte_offset"] if row else 0

    text, new_offset = _transcript_delta(path, offset)
    if len(text) < args.min_delta and not args.force:
        # Watermark intentionally NOT advanced: small deltas accumulate until
        # they're worth one extraction call.
        print(f"delta {len(text)} chars < {args.min_delta} — skipped (accumulating)")
        return 0
    if len(text) > HARVEST_MAX_DELTA:
        text = text[-HARVEST_MAX_DELTA:]

    # Project hint from Claude Code's transcript dir naming (-Users-x-Documents-foo).
    project = args.project or path.parent.name.split("-")[-1] or "general"

    if args.dry_run:
        print(HARVEST_PROMPT.format(project=project, conversation=text))
        print(f"\n[dry-run] {len(text)} chars would be sent to model '{args.model}'")
        return 0

    candidates = _extract_candidates(
        HARVEST_PROMPT.format(project=project, conversation=text), args.model)
    if candidates is None:
        return _err("extraction call failed — watermark unchanged, will retry next run")

    added, noops, queued = [], [], []
    review_path = Path.home() / ".config/brain/harvest-review.jsonl"
    for c in candidates[:5]:
        title = str(c.get("title", "")).strip()
        content = str(c.get("content", "")).strip()
        if not title:
            continue
        ns = argparse.Namespace(
            type=str(c.get("type", "note")), title=title, content=content,
            project=str(c.get("project", project)),
            tags=[str(c.get("tags", ""))] if c.get("tags") else [],
            abstract=str(c.get("abstract", "")) or None,
            anchor=str(c.get("anchor", "")).strip() or None,
            identity=None,
            no_semantic=args.no_semantic, no_embed=False,
        )
        packet = _reconcile_decide(conn, cursor, title, content, ns)
        if packet["suggestion"] == "add":
            rc = cmd_save(_save_ns(
                **{k: v for k, v in vars(ns).items() if k != "no_semantic"},
                source_file=f"harvest:{path.name}"))
            (added if rc == 0 else noops).append(title)
        elif packet["suggestion"] == "noop":
            noops.append(title)
        else:  # update / review — a human or interactive agent should decide
            review_path.parent.mkdir(parents=True, exist_ok=True)
            with open(review_path, "a") as f:
                f.write(json.dumps({"candidate": vars(ns) | {"tags": ns.tags},
                                    "packet": packet}, ensure_ascii=False,
                                   default=str) + "\n")
            queued.append(title)

    cursor.execute(
        "INSERT INTO harvest_state(path, byte_offset, last_run) VALUES (?, ?, datetime('now')) "
        "ON CONFLICT(path) DO UPDATE SET byte_offset = ?, last_run = datetime('now')",
        (str(path), new_offset, new_offset),
    )
    conn.commit()

    print(f"harvest: {len(candidates)} candidate(s) → {len(added)} added, "
          f"{len(noops)} noop, {len(queued)} queued for review"
          + (f" ({review_path})" if queued else ""))
    return 0


def cmd_consolidate(args) -> int:
    """Find near-duplicate clusters; merge with --merge (Letta's sleep-time
    pass, run on demand). Report mode never writes. Merging marks dups
    invalid_at + superseded_by=keeper — content is never destroyed, and the
    agent should `brain update` the keeper first if the dups held unique facts.
    """
    conn = connect(load_vec=True)
    cursor = conn.cursor()

    # ── merge mode ───────────────────────────────────────────────────────
    if args.merge:
        if len(args.merge) < 2:
            return _err("--merge needs KEEPER DUP [DUP...]")
        if not _schema_flags(conn)["invalid_at"]:
            return _err("invalid_at column missing — run `brain migrate` (schema v4)")
        keeper_uid, dup_uids = args.merge[0], args.merge[1:]
        keeper = cursor.execute(
            "SELECT id, uid FROM memories WHERE uid = ? AND deleted_at IS NULL",
            (keeper_uid,)).fetchone()
        if not keeper:
            return _err(f"keeper uid {keeper_uid} not found")
        merged = 0
        for dup in dup_uids:
            if dup == keeper_uid:
                return _err("keeper cannot be its own duplicate")
            row = cursor.execute(
                "SELECT id, invalid_at FROM memories WHERE uid = ? AND deleted_at IS NULL",
                (dup,)).fetchone()
            if not row:
                return _err(f"dup uid {dup} not found")
            if row["invalid_at"] is not None:
                print(f"- {dup} already invalidated, skipped")
                continue
            cursor.execute(
                "UPDATE memories SET invalid_at = datetime('now'), superseded_by = ? WHERE id = ?",
                (keeper["uid"], row["id"]))
            log_alteration(conn, dup, "invalidate",
                           delta=f"superseded_by {keeper['uid']}",
                           reason="consolidate merge")
            try:
                cursor.execute(
                    "INSERT INTO memory_links(src_id, dst_id, kind) VALUES (?, ?, 'superseded_by')",
                    (row["id"], keeper["id"]))
            except sqlite3.IntegrityError:
                pass
            merged += 1
        conn.commit()
        print(f"merged {merged} duplicate(s) into {keeper_uid}")
        return 0

    # ── report mode (read-only) ──────────────────────────────────────────
    filter_sql, filter_params = _build_filters(conn, args)
    rows = cursor.execute(f"""
        SELECT m.id, m.uid, m.title, m.project, m.created_at, m.content_hash
        FROM memories m WHERE m.deleted_at IS NULL{filter_sql}
    """, filter_params).fetchall()
    by_id = {r["id"]: r for r in rows}

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    pair_sim: dict[tuple[int, int], float] = {}

    # Exact-hash duplicates (sim 1.0) — cheap SQL, catches --force accidents.
    for hash_val, ids_csv in cursor.execute(f"""
        SELECT m.content_hash, GROUP_CONCAT(m.id) FROM memories m
        WHERE m.deleted_at IS NULL{filter_sql}
        GROUP BY m.content_hash HAVING COUNT(*) > 1
    """, filter_params):
        ids = [int(i) for i in ids_csv.split(",")]
        for a, b in zip(ids, ids[1:]):
            union(a, b)
            pair_sim[(min(a, b), max(a, b))] = 1.0

    # Embedding near-duplicates: each memory's first chunk KNN'd against all
    # chunks; cross-memory hits above threshold cluster together.
    if _vec_loaded:
        t0 = time.perf_counter()
        for mid in by_id:
            vec = cursor.execute(
                "SELECT embedding FROM memory_chunks WHERE rowid = ?",
                (mid * CHUNK_CAP,)).fetchone()
            if not vec:
                continue
            try:
                for r in cursor.execute("""
                    SELECT memory_id AS id, distance FROM memory_chunks
                    WHERE embedding MATCH ? AND k = 8
                """, (vec["embedding"],)):
                    other, sim = r["id"], 1.0 - r["distance"]
                    if other != mid and other in by_id and sim >= args.threshold:
                        union(mid, other)
                        key = (min(mid, other), max(mid, other))
                        pair_sim[key] = max(pair_sim.get(key, 0.0), sim)
            except sqlite3.OperationalError as e:
                _warn(f"consolidate KNN failed ({e}); hash-only clusters")
                break
        _timing("consolidate KNN sweep", t0)

    clusters: dict[int, list[int]] = {}
    for mid in set(parent) | {m for pair in pair_sim for m in pair}:
        clusters.setdefault(find(mid), []).append(mid)
    clusters = {k: sorted(v) for k, v in clusters.items() if len(v) > 1}

    def cluster_sim(members: list[int]) -> float:
        sims = [s for (a, b), s in pair_sim.items() if a in members and b in members]
        return max(sims) if sims else 0.0

    ranked = sorted(clusters.values(), key=cluster_sim, reverse=True)[:args.limit]
    if not ranked:
        print("no near-duplicate clusters found")
        return 0

    if args.json:
        print(json.dumps([{
            "max_sim": round(cluster_sim(members), 4),
            "members": [{"uid": by_id[m]["uid"], "title": by_id[m]["title"],
                         "project": by_id[m]["project"],
                         "created_at": by_id[m]["created_at"]} for m in members],
        } for members in ranked], ensure_ascii=False, indent=2))
    else:
        print(f"\n{len(ranked)} cluster(s) (threshold {args.threshold}):\n")
        for members in ranked:
            print(f"≈ {cluster_sim(members):.2f}")
            for m in members:
                r = by_id[m]
                proj = f" ({r['project']})" if r["project"] else ""
                print(f"    {r['uid']}  {r['title']}{proj}  {str(r['created_at'] or '')[:10]}")
            print("    → review, move unique facts into the keeper, then:")
            print(f"      brain consolidate --merge {' '.join(by_id[m]['uid'] for m in members)}\n")
    return 0


def cmd_history(args) -> int:
    """Print the alterations log for one memory, chronologically."""
    conn = connect()
    ensure_alterations_table(conn)
    rows = conn.execute(
        "SELECT ts, kind, reason, delta FROM alterations "
        "WHERE memory_uid = ? ORDER BY ts ASC, id ASC",
        (args.uid,),
    ).fetchall()
    if not rows:
        print(f"no alterations recorded for {args.uid}")
        return 0
    for r in rows:
        reason = r["reason"] or ""
        delta = (r["delta"] or "").replace("\n", " ")
        if len(delta) > 80:
            delta = delta[:80] + "…"
        print(f"{r['ts']} | {r['kind']:7} | {reason:24} | {delta}")
    return 0


def cmd_tags(args) -> int:
    conn = connect()
    rows = conn.execute("""
        SELECT name, use_count FROM tags WHERE use_count > 0 ORDER BY use_count DESC LIMIT ?
    """, (args.limit,)).fetchall()
    for r in rows:
        print(f"{r['use_count']:5}  {r['name']}")
    return 0


def cmd_stats(args) -> int:
    conn = connect(load_vec=True)
    total = conn.execute("SELECT COUNT(*) AS n FROM memories WHERE deleted_at IS NULL").fetchone()["n"]
    deleted = conn.execute("SELECT COUNT(*) AS n FROM memories WHERE deleted_at IS NOT NULL").fetchone()["n"]
    by_type = conn.execute("""
        SELECT COALESCE(canonical_type, type) AS t, COUNT(*) AS n
        FROM memories WHERE deleted_at IS NULL GROUP BY t ORDER BY n DESC
    """).fetchall()
    by_proj = conn.execute("""
        SELECT project, COUNT(*) AS n FROM memories
        WHERE deleted_at IS NULL AND project IS NOT NULL AND project != ''
        GROUP BY project ORDER BY n DESC LIMIT 10
    """).fetchall()
    embed_count = 0
    try:
        embed_count = conn.execute(
            "SELECT COUNT(DISTINCT memory_id) AS n FROM memory_chunks"
        ).fetchone()["n"]
    except sqlite3.OperationalError as e:
        _warn(f"memory_chunks count failed ({e}); trying legacy memory_vectors")
        try:
            embed_count = conn.execute("SELECT COUNT(*) AS n FROM memory_vectors").fetchone()["n"]
        except sqlite3.OperationalError as e2:
            _warn(f"memory_vectors count failed: {e2}")
    schema_v = conn.execute("SELECT value FROM stats WHERE key = 'brain_schema_version'").fetchone()

    invalidated = 0
    if _schema_flags(conn)["invalid_at"]:
        invalidated = conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE deleted_at IS NULL AND invalid_at IS NOT NULL"
        ).fetchone()["n"]

    print(f"\nbrain v{schema_v['value'] if schema_v else '?'}")
    print(f"  active:    {total - invalidated}")
    print(f"  invalid:   {invalidated}")
    print(f"  deleted:   {deleted}")
    print(f"  embedded:  {embed_count}/{total}")
    print("\nby type:")
    for r in by_type:
        print(f"  {r['t']:12} {r['n']}")
    print("\ntop projects:")
    for r in by_proj:
        print(f"  {r['project']:30} {r['n']}")
    return 0


def cmd_recent(args) -> int:
    conn = connect()
    extra = " AND invalid_at IS NULL" if _schema_flags(conn)["invalid_at"] else ""
    rows = conn.execute(f"""
        SELECT uid, COALESCE(canonical_type, type) AS type, title, project, created_at
        FROM memories WHERE deleted_at IS NULL{extra}
        ORDER BY created_at DESC LIMIT ?
    """, (args.n,)).fetchall()
    for r in rows:
        proj = f" ({r['project']})" if r["project"] else ""
        print(f"[{r['type']:8}] {r['title']}{proj}  {(r['created_at'] or '')[:10]}  {r['uid']}")
    return 0


def cmd_reindex(args) -> int:
    """Backfill chunked embeddings. Heavy — uses sentence-transformers.

    Default: only memories with no chunks yet. --full: re-embed everything
    (use after changing chunking parameters or the model).
    """
    conn = connect(load_vec=True)
    if not _vec_loaded:
        return _err("sqlite-vec not loaded. Re-run via uv to install: `uv run brain.py reindex`")

    model = get_embedder()
    if model is None:
        return _err("sentence-transformers not installed. Add it to inline deps and re-run via uv.")

    # Idempotent — vec extension is loaded, so create the chunk table if absent.
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks USING vec0(
            memory_id INTEGER,
            embedding FLOAT[384] distance_metric=cosine
        )
    """)
    conn.commit()

    rows = conn.execute(
        f"SELECT id, title, content, {_anchor_col(conn).replace('m.', '')} "
        "FROM memories WHERE deleted_at IS NULL"
    ).fetchall()
    if not args.full:
        done = {r[0] for r in conn.execute("SELECT DISTINCT memory_id FROM memory_chunks")}
        rows = [r for r in rows if r["id"] not in done]

    print(f"→ embedding {len(rows)} memories (chunked)…")
    for i, r in enumerate(rows, 1):
        embed_memory(conn, r["id"], r["title"] or "", r["content"] or "",
                     _row_get(r, "anchor"))
        conn.commit()  # embed_memory no longer commits; keep reindex incremental
        if i % 50 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")
    print("✓ reindex complete")
    return 0


def cmd_doctor(args) -> int:
    """Index health check: missing vectors, orphans, FTS corruption, WAL.

    Exists because the pre-atomic save flow committed the memory row and its
    chunk vectors separately — a crash in between left memories silently
    invisible to semantic search (13 found in prod). --fix re-embeds missing
    vectors, deletes orphan rows, and rebuilds FTS if its check fails.
    """
    conn = connect(load_vec=True)
    cursor = conn.cursor()
    problems = 0

    def table_exists(name: str) -> bool:
        return cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ?", (name,)
        ).fetchone() is not None

    active = {r["id"] for r in cursor.execute(
        "SELECT id FROM memories WHERE deleted_at IS NULL")}
    has_chunks = _vec_loaded and table_exists("memory_chunks")
    has_vectors = _vec_loaded and table_exists("memory_vectors")
    chunked: set[int] = set()
    if has_chunks:
        chunked = {r[0] for r in cursor.execute("SELECT DISTINCT memory_id FROM memory_chunks")}

    # (a) active memories with zero chunk rows → invisible to semantic search.
    missing: list[int] = []
    if has_chunks:
        missing = sorted(active - chunked)
        if missing:
            problems += len(missing)
            print(f"✗ {len(missing)} active memories have no chunk vectors (semantic-invisible)")
        else:
            print(f"✓ vectors: all {len(active)} active memories have chunks")
    else:
        print("- vectors: check skipped (sqlite-vec or memory_chunks unavailable)")

    # (b) orphan chunk/vector rows — memory deleted or missing. Harmless to
    #     results (search re-filters) but orphan chunks waste KNN slots.
    #     Legacy memory_vectors orphans only matter when the legacy table is
    #     actually read (no memory_chunks) — otherwise informational, like (d).
    orphan_chunks: list[int] = []
    orphan_vectors: list[int] = []
    if has_chunks:
        orphan_chunks = [r["rowid"] for r in cursor.execute(
            "SELECT rowid, memory_id FROM memory_chunks") if r["memory_id"] not in active]
    if has_vectors:
        orphan_vectors = [r["memory_id"] for r in cursor.execute(
            "SELECT memory_id FROM memory_vectors") if r["memory_id"] not in active]
    if orphan_chunks or (orphan_vectors and not has_chunks):
        problems += len(orphan_chunks) + (0 if has_chunks else len(orphan_vectors))
        print(f"✗ orphans: {len(orphan_chunks)} chunk rows, "
              f"{len(orphan_vectors)} legacy vector rows point at deleted/missing memories")
    elif orphan_vectors:
        print(f"i orphans: {len(orphan_vectors)} legacy vector rows point at "
              "deleted/missing memories (never read while chunks exist; --fix prunes)")
    elif has_chunks or has_vectors:
        print("✓ orphans: none")

    # (c) FTS external-content integrity (rank=1 verifies against memories).
    fts_ok = True
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts, rank) VALUES('integrity-check', 1)")
        print("✓ FTS: index consistent with memories")
    except sqlite3.DatabaseError as e:
        fts_ok = False
        problems += 1
        print(f"✗ FTS: integrity check failed ({e})")

    # (d) stale legacy vectors shadowed by chunks — informational only.
    if has_vectors and chunked:
        stale = sum(1 for r in cursor.execute("SELECT memory_id FROM memory_vectors")
                    if r["memory_id"] in chunked)
        if stale:
            print(f"i legacy: {stale} memory_vectors rows shadowed by chunks (harmless, never read)")

    # (e) journal mode + schema version.
    mode = cursor.execute("PRAGMA journal_mode").fetchone()[0]
    try:
        sv = cursor.execute(
            "SELECT value FROM stats WHERE key = 'brain_schema_version'").fetchone()
        version = sv["value"] if sv else "?"
    except sqlite3.OperationalError as e:
        _warn(f"stats table missing: {e}")
        version = "?"
    print(f"i journal_mode={mode} schema_version={version}")

    if not args.fix:
        if problems:
            print(f"\n{problems} problem(s) found — run `brain doctor --fix`")
            return 1
        print("\nhealthy")
        return 0

    # ── --fix ────────────────────────────────────────────────────────────
    unfixed = 0
    if orphan_chunks:
        for rid in orphan_chunks:
            cursor.execute("DELETE FROM memory_chunks WHERE rowid = ?", (rid,))
        conn.commit()
        print(f"→ deleted {len(orphan_chunks)} orphan chunk rows")
    if orphan_vectors:
        for mid in orphan_vectors:
            cursor.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (mid,))
        conn.commit()
        print(f"→ deleted {len(orphan_vectors)} orphan vector rows")
    if missing:
        if get_embedder() is None:
            unfixed += len(missing)
            print("⚠ cannot re-embed: sentence-transformers unavailable (FTS-only mode)")
        else:
            ok = 0
            for mid in missing:
                r = cursor.execute(
                    f"SELECT title, content, {_anchor_col(conn).replace('m.', '')} "
                    "FROM memories WHERE id = ?", (mid,)).fetchone()
                if r and embed_memory(conn, mid, r["title"] or "", r["content"] or "",
                                      _row_get(r, "anchor")):
                    conn.commit()
                    ok += 1
                else:
                    unfixed += 1
            print(f"→ re-embedded {ok}/{len(missing)} missing memories")
    if not fts_ok:
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            conn.commit()
            print("→ FTS index rebuilt")
        except sqlite3.DatabaseError as e:
            unfixed += 1
            print(f"✗ FTS rebuild failed: {e}")
    if unfixed:
        print(f"\n{unfixed} problem(s) NOT fixed")
        return 1
    print("\nall fixable problems repaired")
    return 0


def cmd_migrate(args) -> int:
    """Run migration. Calls the migrate.py sibling script."""
    from subprocess import run
    here = Path(__file__).resolve().parent
    return run([sys.executable, str(here / "scripts" / "migrate.py")]).returncode


# Backwards-compat aliases for old `brain` script users.
def cmd_query(args) -> int:
    args.limit = 10
    args.type = None
    args.project = None
    args.since_days = None
    args.no_semantic = False
    args.json = False
    return cmd_search(args)


def cmd_add(args) -> int:
    """Old-style: --type=X --title=X --content=X. Maps to save."""
    return cmd_save(args)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _err(msg: str) -> int:
    print(f"✗ {msg}", file=sys.stderr)
    return 1


def _tags_for(conn: sqlite3.Connection, memory_id: int) -> list[str]:
    rows = conn.execute("""
        SELECT t.name FROM memory_tags mt JOIN tags t ON t.id = mt.tag_id
        WHERE mt.memory_id = ? ORDER BY t.use_count DESC
    """, (memory_id,)).fetchall()
    return [r["name"] for r in rows]


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="brain",
        description="Personal knowledge CLI for ~/brain.db",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # save
    s = sub.add_parser("save", help="Save a new memory")
    s.add_argument("--type", required=True, help="learning|decision|bug|snippet|note|task|person|project (aliases ok)")
    s.add_argument("--title", required=True)
    s.add_argument("--content", default="")
    s.add_argument("--project", default="")
    s.add_argument("--area", default="")
    s.add_argument("--tags", action="append", default=[], help="comma-separated; can repeat flag")
    s.add_argument("--source-file", default="")
    s.add_argument("--abstract", help="~1-sentence L0 summary used by `brain context`")
    s.add_argument("--identity", help="declared merge key (e.g. \"corena deploy runbook\") — "
                                      "at most one live memory may hold it; reconcile merges on it")
    s.add_argument("--anchor", help="entity this is about (client/repo/person/system) — "
                                    "groups lessons in `brain context`")
    s.add_argument("--force", action="store_true", help="bypass dedup")
    s.add_argument("--no-embed", action="store_true", help="skip embedding")
    # task-only:
    s.add_argument("--status", choices=["pending", "doing", "done", "waiting", "cancelled"])
    s.add_argument("--priority", choices=["p1", "p2", "p3", "p4"])
    s.add_argument("--energy", choices=["high", "medium", "low"])
    s.add_argument("--points", type=int)
    s.add_argument("--due-at")
    s.add_argument("--external-ref")
    s.add_argument("--parent-uid")
    s.set_defaults(func=cmd_save)

    # search
    s = sub.add_parser("search", help="Hybrid search (BM25 + semantic + decay)")
    s.add_argument("query")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--type", action="append", help="filter by canonical type (repeatable)")
    s.add_argument("--project")
    s.add_argument("--since-days", type=int, help="only memories from last N days")
    s.add_argument("--no-semantic", action="store_true")
    s.add_argument("--explain", action="store_true", help="show per-result score decomposition")
    s.add_argument("--compact", action="store_true", help="uid+title only (~10x fewer tokens/hit)")
    s.add_argument("--include-invalid", action="store_true", help="also show invalidated memories")
    s.add_argument("--as-of", help="time travel: what was true at this date (YYYY-MM-DD)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    # context
    s = sub.add_parser("context", help="Prompt-ready memory block under a hard token budget")
    s.add_argument("query", help="topic seed (project name, current task, ...)")
    s.add_argument("--budget", type=int, default=2000, help="max tokens to emit (default 2000)")
    s.add_argument("--limit", type=int, default=20, help="retrieval pool before budgeting")
    s.add_argument("--type", action="append")
    s.add_argument("--project")
    s.add_argument("--since-days", type=int)
    s.add_argument("--no-semantic", action="store_true")
    s.add_argument("--no-blocks", action="store_true", help="omit pinned core blocks")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_context)

    # artifacts
    s = sub.add_parser("artifact",
                       help="Track the real files memories point at (pointer + drift detection)")
    s.add_argument("action", choices=["scan", "check", "list"], nargs="?", default="list")
    s.add_argument("--missing", action="store_true", help="list only: vanished files")
    s.add_argument("--large", action="store_true", help="list only: >=200KB")
    s.add_argument("--limit", type=int, default=25)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(func=cmd_artifact)

    # secrets
    s = sub.add_parser("secrets",
                       help="Move secret values out of memory content into a chmod-600 env file")
    s.add_argument("--scan", action="store_true", help="report only (default if no --extract)")
    s.add_argument("--extract", action="store_true", help="rewrite content to ${VAR} references")
    s.add_argument("--dry-run", action="store_true", help="with --extract: show the plan")
    s.add_argument("--env-file", help=f"default {SECRETS_ENV_DEFAULT}")
    s.add_argument("--no-embed", action="store_true")
    s.set_defaults(func=cmd_secrets)

    # anchor backfill
    s = sub.add_parser("anchor", help="Backfill the `anchor` entity on memories that lack one")
    s.add_argument("--limit", type=int, help="only the N most recent unanchored")
    s.add_argument("--model", default="sonnet")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--no-embed", action="store_true", help="skip re-embedding (FTS still updates)")
    s.set_defaults(func=cmd_anchor)

    # feedback
    s = sub.add_parser("feedback",
                       help="Mark a memory as useful/misleading — feeds search ranking")
    s.add_argument("uid")
    s.add_argument("signal", choices=["up", "down"])
    s.add_argument("--note", help="why (stored, not used in ranking)")
    s.set_defaults(func=cmd_feedback)

    # block (pinned core context)
    s = sub.add_parser("block", help="Pinned core blocks injected into every `brain context`")
    s.add_argument("action", choices=["set", "list", "rm"], nargs="?", default="list")
    s.add_argument("label", nargs="?")
    s.add_argument("value", nargs="?")
    s.add_argument("--char-limit", type=int, default=400)
    s.add_argument("--position", type=int, default=100, help="sort order (lower = first)")
    s.set_defaults(func=cmd_block)

    # review (harvest queue triage)
    s = sub.add_parser("review", help="Triage the harvest review queue (ambiguous candidates)")
    s.add_argument("--limit", type=int, default=10, help="packets to show (default 10)")
    s.add_argument("--auto", action="store_true",
                   help="resolve only the unambiguous ends (drop near-dups, save clearly-new)")
    s.add_argument("--judge", action="store_true",
                   help="with --auto: add an LLM durability pass — drops transient junk "
                        "that similarity cannot detect (~27%% of otherwise-savable candidates)")
    s.add_argument("--model", default="sonnet", help="model for --judge (default sonnet)")
    s.add_argument("--dry-run", action="store_true", help="with --auto: report, change nothing")
    s.add_argument("--resolve", type=int, metavar="N", help="resolve queue entry N")
    s.add_argument("--action", choices=["drop", "save", "update"], help="with --resolve")
    s.add_argument("--uid", help="with --resolve --action update: merge target")
    s.add_argument("--clear", action="store_true", help="drop the whole queue (needs --yes)")
    s.add_argument("--yes", action="store_true")
    s.add_argument("--file", help="queue path (default ~/.config/brain/harvest-review.jsonl)")
    s.add_argument("--no-semantic", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_review)

    # reconcile
    s = sub.add_parser("reconcile",
                       help="ADD/UPDATE/NOOP decision packet for a candidate fact (use before save)")
    s.add_argument("--type", default="note")
    s.add_argument("--title", required=True)
    s.add_argument("--content", default="")
    s.add_argument("--project", default="")
    s.add_argument("--tags", action="append", default=[])
    s.add_argument("--abstract")
    s.add_argument("--identity", help="declared merge key — an existing holder forces 'update'")
    s.add_argument("--anchor", help="entity this fact is about")
    s.add_argument("--auto", action="store_true",
                   help="apply 'add' directly; exit 2 on noop, 3 when the agent must decide")
    s.add_argument("--no-embed", action="store_true")
    s.add_argument("--no-semantic", action="store_true")
    s.set_defaults(func=cmd_reconcile)

    # invalidate
    s = sub.add_parser("invalidate",
                       help="Mark a memory's fact as no longer true (soft, reversible)")
    s.add_argument("uid")
    s.add_argument("--reason", help="why (recorded in alterations)")
    s.add_argument("--superseded-by", help="uid of the memory that replaces this fact")
    s.add_argument("--undo", action="store_true", help="restore a previously invalidated memory")
    s.set_defaults(func=cmd_invalidate)

    # harvest
    s = sub.add_parser("harvest",
                       help="Extract memories from an agent transcript (JSONL) via one LLM call")
    s.add_argument("transcript", help="path to a Claude Code session .jsonl")
    s.add_argument("--model", default="haiku", help="model for the extraction call (default haiku)")
    s.add_argument("--min-delta", type=int, default=2000,
                   help="min chars of new conversation before extracting (default 2000)")
    s.add_argument("--project", help="override the project slug inferred from the path")
    s.add_argument("--force", action="store_true", help="extract even below --min-delta")
    s.add_argument("--dry-run", action="store_true", help="show what would be sent, change nothing")
    s.add_argument("--no-semantic", action="store_true")
    s.set_defaults(func=cmd_harvest)

    # consolidate
    s = sub.add_parser("consolidate",
                       help="Find near-duplicate clusters (read-only); --merge to supersede dups")
    s.add_argument("--threshold", type=float, default=0.85, help="cosine sim cutoff (default 0.85)")
    s.add_argument("--limit", type=int, default=20, help="max clusters to report")
    s.add_argument("--project")
    s.add_argument("--merge", nargs="+", metavar=("KEEPER", "DUP"),
                   help="invalidate DUPs as superseded by KEEPER")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_consolidate)

    # get
    s = sub.add_parser("get", help="Show full memory by uid")
    s.add_argument("uid")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_get)

    # link
    s = sub.add_parser("link", help="Link two memories")
    s.add_argument("src", help="source uid")
    s.add_argument("dst", help="destination uid")
    s.add_argument("kind", help="cites|caused_by|fixed_by|superseded_by|blocks|related_to|duplicate_of")
    s.set_defaults(func=cmd_link)

    # delete / restore
    s = sub.add_parser("delete", help="Soft-delete a memory")
    s.add_argument("uid")
    s.set_defaults(func=cmd_delete)
    s = sub.add_parser("restore", help="Restore a soft-deleted memory")
    s.add_argument("uid")
    s.set_defaults(func=cmd_restore)

    # update
    s = sub.add_parser("update", help="Append to or replace a memory's content (merge)")
    s.add_argument("uid")
    s.add_argument("--append", help="append a dated bullet to content")
    s.add_argument("--replace", help="replace the full content")
    s.add_argument("--abstract", help="replace the L0 abstract used by `brain context`")
    s.add_argument("--reason", help="why this change (recorded in alterations)")
    s.set_defaults(func=cmd_update)

    # history
    s = sub.add_parser("history", help="Show the alterations log for a memory")
    s.add_argument("uid")
    s.set_defaults(func=cmd_history)

    # tags
    s = sub.add_parser("tags", help="List tags by use count")
    s.add_argument("--limit", type=int, default=30)
    s.set_defaults(func=cmd_tags)

    # stats / recent
    s = sub.add_parser("stats", help="Show DB stats")
    s.set_defaults(func=cmd_stats)
    s = sub.add_parser("recent", help="Show N most recent memories")
    s.add_argument("n", type=int, nargs="?", default=10)
    s.set_defaults(func=cmd_recent)

    # reindex
    s = sub.add_parser("reindex", help="Backfill chunked embeddings for memories that lack them")
    s.add_argument("--full", action="store_true", help="re-embed ALL memories, not just missing ones")
    s.set_defaults(func=cmd_reindex)

    # doctor
    s = sub.add_parser("doctor", help="Check index health (missing vectors, orphans, FTS)")
    s.add_argument("--fix", action="store_true",
                   help="re-embed missing, delete orphans, rebuild FTS if corrupt")
    s.set_defaults(func=cmd_doctor)

    # migrate
    s = sub.add_parser("migrate", help="Apply v3 schema migration")
    s.set_defaults(func=cmd_migrate)

    # backwards compat
    s = sub.add_parser("query", help="(legacy) alias for `search`")
    s.add_argument("query")
    s.set_defaults(func=cmd_query)
    s = sub.add_parser("add", help="(legacy) alias for `save`")
    s.add_argument("--type", required=True)
    s.add_argument("--title", required=True)
    s.add_argument("--content", default="")
    s.add_argument("--project", default="")
    s.add_argument("--area", default="")
    s.add_argument("--tags", action="append", default=[])
    s.add_argument("--source-file", default="")
    s.add_argument("--force", action="store_true")
    s.add_argument("--no-embed", action="store_true")
    s.add_argument("--status", choices=["pending", "doing", "done", "waiting", "cancelled"])
    s.add_argument("--priority", choices=["p1", "p2", "p3", "p4"])
    s.add_argument("--energy", choices=["high", "medium", "low"])
    s.add_argument("--points", type=int)
    s.add_argument("--due-at")
    s.add_argument("--external-ref")
    s.add_argument("--parent-uid")
    s.set_defaults(func=cmd_add)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
