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


def chunk_texts(title: str, content: str) -> list[str]:
    """Embeddable texts: title prepended to every chunk so each stays topical."""
    return [f"{title}\n{c}" if c else title for c in _split_chunks(content)][:CHUNK_CAP]


def embed_memory(conn: sqlite3.Connection, memory_id: int, title: str, content: str) -> bool:
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
    texts = chunk_texts(title, content)
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


def _embed_in_txn(conn, cursor, memory_id: int, title: str, content: str, uid: str) -> None:
    """Embed inside the caller's OPEN transaction (savepoint-guarded).

    Atomicity: memory row + chunks commit together — no crash window where a
    committed memory has no vectors. Resilience: an embedding exception rolls
    back only the savepoint, so the save itself still commits (FTS-only mode).
    """
    cursor.execute("SAVEPOINT embed")
    try:
        embed_memory(conn, memory_id, title, content)
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

    # Audit trail: record creation.
    log_alteration(conn, uid, "create", delta=title, reason=None)

    # Embed in the SAME transaction, then one commit (see _embed_in_txn).
    if not args.no_embed and _vec_loaded:
        _embed_in_txn(conn, cursor, memory_id, title, content, uid)

    conn.commit()

    print(uid)
    return 0


def _build_filters(conn: sqlite3.Connection, args) -> tuple[str, list]:
    """Shared --type/--project/--since-days filter SQL (alias `m`)."""
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
    return sql, params


def _explain_line(e: dict) -> str:
    """One compact human line: ranks, best-chunk sim, rrf parts, bonuses."""
    fts = f"#{e['fts_rank'] + 1}" if e["fts_rank"] is not None else "-"
    sem = f"#{e['sem_rank'] + 1}" if e["sem_rank"] is not None else "-"
    sim = f"{e['sim']:.3f}" if e["sim"] is not None else "-"
    return (f"fts={fts} sem={sem} sim={sim} "
            f"rrf={e['rrf_fts']:.3f}+{e['rrf_sem']:.3f} "
            f"rec=+{e['recency_bonus']:.3f} acc=+{e['access_bonus']:.3f} "
            f"= {e['final']:.3f}")


def _print_results(top: list, conn: sqlite3.Connection, query: str, as_json: bool,
                   explain: dict[int, dict] | None = None) -> None:
    if as_json:
        out = []
        for score, r in top:
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
        _print_results([(0.0, r) for r in rows], conn, "(listing)", args.json)
        return 0

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
    if _vec_loaded and not args.no_semantic:
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
        if args.json:
            print("[]")
        else:
            print(f"No results for: {query}")
        return 0

    placeholders = ",".join("?" * len(candidate_ids))
    rows = cursor.execute(f"""
        SELECT m.id, m.uid, m.canonical_type AS type, m.title, m.content, m.project,
               m.created_at, m.access_count
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
    sims = dict(sem_pairs)  # best-chunk cosine sim per memory (for --explain)
    explain: dict[int, dict] | None = {} if getattr(args, "explain", False) else None
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    full = 2.0 / (RRF_K + 1)
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
        score = rrf_fts + rrf_sem + recency + access
        if explain is not None:
            explain[mid] = {
                "fts_rank": fts_rank.get(mid),
                "sem_rank": sem_rank.get(mid),
                "sim": round(sims[mid], 4) if mid in sims else None,
                "rrf_fts": round(rrf_fts, 4), "rrf_sem": round(rrf_sem, 4),
                "rrf": round(rrf_fts + rrf_sem, 4),
                "recency_bonus": round(recency, 4), "access_bonus": round(access, 4),
                "final": round(score, 4),
            }
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]
    _timing("fusion", t_fusion)
    _timing("total", t_total)

    # access_count is bumped on `brain get` only — search must not reinforce
    # its own ranking (rich-get-richer loop on frequently-surfaced junk).
    _print_results(top, conn, query, args.json, explain=explain)
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

    # Update access tracking.
    conn.execute("UPDATE memories SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?", (r["id"],))
    conn.commit()

    if args.json:
        d = dict(r)
        d.pop("content_hash", None)
        print(json.dumps(d, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"\n{r['title']}")
        print(f"[{r['canonical_type'] or r['type']}] {r['uid']} · {r['created_at']}")
        if r["project"]:
            print(f"project: {r['project']}")
        if r["all_tags"]:
            print(f"tags: {r['all_tags']}")
        print(f"\n{r['content'] or ''}\n")
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
    if not args.append and args.replace is None:
        return _err("provide --append \"<text>\" or --replace \"<full content>\"")
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

    if args.append:
        text = args.append.strip()
        if not text:
            return _err("--append text must not be empty")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bullet = f"\n\n- [{today}] {text}"
        new_content = old_content + bullet
        kind = "append"
        delta = bullet.strip()
    else:
        new_content = args.replace
        kind = "replace"
        delta = new_content

    new_hash = content_hash(title, new_content)

    # The AFTER UPDATE trigger on memories re-syncs memories_fts automatically,
    # and a BEFORE UPDATE trigger snapshots the old content into memory_versions.
    cursor.execute(
        "UPDATE memories SET content = ?, content_hash = ?, updated_at = datetime('now') "
        "WHERE id = ?",
        (new_content, new_hash, memory_id),
    )

    log_alteration(conn, args.uid, kind, delta=delta, reason=args.reason)

    # Re-embed in the SAME transaction, then one commit (see _embed_in_txn).
    if _vec_loaded:
        _embed_in_txn(conn, cursor, memory_id, title, new_content, args.uid)

    conn.commit()

    print(args.uid)
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

    print(f"\nbrain v{schema_v['value'] if schema_v else '?'}")
    print(f"  active:    {total}")
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
    rows = conn.execute("""
        SELECT uid, COALESCE(canonical_type, type) AS type, title, project, created_at
        FROM memories WHERE deleted_at IS NULL
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
        "SELECT id, title, content FROM memories WHERE deleted_at IS NULL"
    ).fetchall()
    if not args.full:
        done = {r[0] for r in conn.execute("SELECT DISTINCT memory_id FROM memory_chunks")}
        rows = [r for r in rows if r["id"] not in done]

    print(f"→ embedding {len(rows)} memories (chunked)…")
    for i, r in enumerate(rows, 1):
        embed_memory(conn, r["id"], r["title"] or "", r["content"] or "")
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
                    "SELECT title, content FROM memories WHERE id = ?", (mid,)).fetchone()
                if r and embed_memory(conn, mid, r["title"] or "", r["content"] or ""):
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
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

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
