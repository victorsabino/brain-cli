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
import secrets
import sqlite3
import sys
from base64 import b32encode
from datetime import datetime, timezone
from pathlib import Path

DB = Path.home() / "brain.db"

CANONICAL_TYPES = {"learning", "decision", "bug", "snippet", "note", "task", "person", "project"}
LINK_KINDS = {"cites", "caused_by", "fixed_by", "superseded_by", "blocks", "related_to", "duplicate_of"}

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
# Embeddings (lazy)
# ────────────────────────────────────────────────────────────────────────────


def get_embedder():
    """Lazy-load sentence-transformers. Returns None if not installed."""
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        # Multilingual, 384-dim, ~470MB, handles PT/EN well.
        _embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return _embed_model
    except ImportError:
        return None


def embed(text: str) -> bytes | None:
    model = get_embedder()
    if model is None:
        return None
    vec = model.encode(text, normalize_embeddings=True)
    return vec.astype("float32").tobytes()


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

    conn.commit()

    # Embedding (after commit so save still succeeds if model fails).
    if not args.no_embed and _vec_loaded:
        vec = embed(f"{title}\n{content}")
        if vec is not None:
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO memory_vectors(memory_id, embedding) VALUES (?, ?)",
                    (memory_id, vec),
                )
                conn.commit()
            except sqlite3.OperationalError:
                pass

    print(uid)
    return 0


def cmd_search(args) -> int:
    conn = connect(load_vec=True)
    cursor = conn.cursor()

    query = args.query
    limit = args.limit

    # 1. FTS5 keyword (always available).
    fts_terms = " OR ".join(f'"{w}"' for w in query.split() if len(w) > 1)
    fts_hits: dict[int, float] = {}
    if fts_terms:
        try:
            for r in cursor.execute("""
                SELECT m.rowid AS id, bm25(memories_fts) AS score
                FROM memories_fts
                JOIN memories m ON m.rowid = memories_fts.rowid
                WHERE memories_fts MATCH ? AND m.deleted_at IS NULL
                LIMIT ?
            """, (fts_terms, limit * 3)):
                # bm25: lower = better. Invert so higher = better, normalize ~ [0..1].
                fts_hits[r["id"]] = 1.0 / (1.0 + max(r["score"], 0))
        except sqlite3.OperationalError:
            pass

    # 2. Semantic (only if vec0 loaded and embedder available).
    sem_hits: dict[int, float] = {}
    if _vec_loaded and not args.no_semantic:
        qvec = embed(query)
        if qvec is not None:
            try:
                for r in cursor.execute("""
                    SELECT memory_id AS id, distance
                    FROM memory_vectors
                    WHERE embedding MATCH ? AND k = ?
                """, (qvec, limit * 3)):
                    sem_hits[r["id"]] = max(0.0, 1.0 - r["distance"])
            except sqlite3.OperationalError:
                pass

    # 3. Merge scores with weights + recency + access boosts.
    candidate_ids = set(fts_hits) | set(sem_hits)
    if not candidate_ids:
        if args.json:
            print("[]")
        else:
            print(f"No results for: {query}")
        return 0

    # Apply filters in SQL for speed.
    placeholders = ",".join("?" * len(candidate_ids))
    rows_sql = f"""
        SELECT m.id, m.uid, m.canonical_type AS type, m.title, m.content, m.project,
               m.created_at, m.last_accessed_at, m.access_count
        FROM memories m
        WHERE m.id IN ({placeholders}) AND m.deleted_at IS NULL
    """
    params: list = list(candidate_ids)
    if args.type:
        rows_sql += f" AND m.canonical_type IN ({','.join('?'*len(args.type))})"
        params.extend(canonical_type(conn, t) for t in args.type)
    if args.project:
        rows_sql += " AND m.project = ?"
        params.append(args.project)
    if args.since_days:
        rows_sql += " AND m.created_at > datetime('now', ?)"
        params.append(f"-{args.since_days} days")

    rows = cursor.execute(rows_sql, params).fetchall()

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    scored = []
    for r in rows:
        fts_score = fts_hits.get(r["id"], 0.0)
        sem_score = sem_hits.get(r["id"], 0.0)
        # Combined score: 60% semantic, 40% keyword.
        base = 0.6 * sem_score + 0.4 * fts_score if sem_score else fts_score
        # Recency: half-life 365 days.
        try:
            age_days = (now - datetime.fromisoformat((r["created_at"] or "").split(".")[0])).days
        except (TypeError, ValueError):
            age_days = 9999
        recency = math.exp(-max(age_days, 0) / 365.0)
        # Access boost: log(count+1).
        access = 1.0 + math.log1p(r["access_count"] or 0)
        final_score = base * recency * access
        scored.append((final_score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:limit]

    # Update access_count + last_accessed_at on the hits.
    if top:
        ids = [r[1]["id"] for r in top]
        cursor.execute(f"""
            UPDATE memories
            SET access_count = access_count + 1, last_accessed_at = datetime('now')
            WHERE id IN ({','.join('?'*len(ids))})
        """, ids)
        conn.commit()

    if args.json:
        out = []
        for score, r in top:
            out.append({
                "uid": r["uid"], "type": r["type"], "title": r["title"],
                "snippet": (r["content"] or "")[:300],
                "project": r["project"], "score": round(score, 3),
                "created_at": r["created_at"],
            })
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
            print(f"           {date} · {r['uid']} · score={score:.2f}{tag_str}\n")
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
        return _err(f"link already exists")


def cmd_delete(args) -> int:
    conn = connect()
    cursor = conn.execute(
        "UPDATE memories SET deleted_at = datetime('now') WHERE uid = ? AND deleted_at IS NULL",
        (args.uid,),
    )
    conn.commit()
    if cursor.rowcount == 0:
        return _err(f"uid {args.uid} not found or already deleted")
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
        embed_count = conn.execute("SELECT COUNT(*) AS n FROM memory_vectors").fetchone()["n"]
    except sqlite3.OperationalError:
        pass
    schema_v = conn.execute("SELECT value FROM stats WHERE key = 'brain_schema_version'").fetchone()

    print(f"\nbrain v{schema_v['value'] if schema_v else '?'}")
    print(f"  active:    {total}")
    print(f"  deleted:   {deleted}")
    print(f"  embedded:  {embed_count}/{total}")
    print(f"\nby type:")
    for r in by_type:
        print(f"  {r['t']:12} {r['n']}")
    print(f"\ntop projects:")
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
    """Backfill embeddings for memories missing them. Heavy — uses sentence-transformers."""
    conn = connect(load_vec=True)
    if not _vec_loaded:
        return _err("sqlite-vec not loaded. Re-run via uv to install: `uv run brain.py reindex`")

    model = get_embedder()
    if model is None:
        return _err("sentence-transformers not installed. Add it to inline deps and re-run via uv.")

    rows = conn.execute("""
        SELECT m.id, m.title, m.content
        FROM memories m
        LEFT JOIN memory_vectors v ON v.memory_id = m.id
        WHERE m.deleted_at IS NULL AND v.memory_id IS NULL
    """).fetchall()
    print(f"→ embedding {len(rows)} memories…")
    batch_size = 32
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        texts = [f"{r['title']}\n{r['content'] or ''}" for r in batch]
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        for r, vec in zip(batch, vecs):
            conn.execute(
                "INSERT OR REPLACE INTO memory_vectors(memory_id, embedding) VALUES (?, ?)",
                (r["id"], vec.astype("float32").tobytes()),
            )
        conn.commit()
        print(f"  {min(i + batch_size, len(rows))}/{len(rows)}")
    print("✓ reindex complete")
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
    s = sub.add_parser("reindex", help="Backfill embeddings for memories that lack them")
    s.set_defaults(func=cmd_reindex)

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
