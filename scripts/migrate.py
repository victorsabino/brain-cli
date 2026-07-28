#!/usr/bin/env python3
"""brain v3 migration runner. Idempotent. Safe to re-run."""

from __future__ import annotations
import argparse
import hashlib
import os
import sqlite3
import sys
import time
from pathlib import Path

DB = Path(os.environ.get("BRAIN_DB") or (Path.home() / "brain.db"))
HERE = Path(__file__).resolve().parent

# Version this script produces. A DB reporting a HIGHER version was written by
# a newer brain — running an old migrator against it could silently downgrade.
# v4 (2026-06): invalid_at (soft fact invalidation), abstract (L0 tier for
# token-budgeted context injection), recall_log (re-extraction guard).
# v5 (2026-07): identity_key (declared merge key — exact, not similarity),
# anchor (the entity a lesson is about), memory_feedback (usefulness signal
# feeding ranking), blocks (pinned always-injected context).
SCHEMA_VERSION = 7

# Tables created by SCHEMA_SQL / setup_vec_extension — used for dry-run planning.
SCHEMA_TABLES = [
    "tags", "memory_tags", "type_aliases", "task_meta",
    "memory_links", "memory_versions", "alterations", "recall_log",
    "memory_feedback", "blocks", "artifacts", "memory_artifacts",
]
VEC_TABLES = ["memory_vectors", "memory_chunks"]

# All ALTER TABLE statements run individually with duplicate-tolerance.
NEW_COLUMNS = [
    ("content_hash", "TEXT"),
    ("deleted_at", "DATETIME"),
    ("last_accessed_at", "DATETIME"),
    ("access_count", "INTEGER NOT NULL DEFAULT 0"),
    ("version", "INTEGER NOT NULL DEFAULT 1"),
    ("superseded_by", "TEXT"),
    ("canonical_type", "TEXT"),
    # v4: NULL = fact still true. Set = superseded/contradicted at that time;
    # default queries filter these out, --as-of / --include-invalid see them.
    ("invalid_at", "DATETIME"),
    # v4: L0 tier — ~1-2 sentence abstract for token-budgeted `brain context`.
    ("abstract", "TEXT"),
    # v5: declared identity — "<type>:<sha256(normalized value)[:16]>". When
    # set, reconcile treats a match as the SAME fact by declaration and merges
    # instead of adding, regardless of embedding similarity.
    ("identity_key", "TEXT"),
    # v5: the entity a lesson is about (client, repo, person, system). Lets
    # `brain context` group entity-anchored lessons instead of emitting a flat
    # list of unrelated facts.
    ("anchor", "TEXT"),
]

# Everything below is pure CREATE TABLE/INDEX/TRIGGER IF NOT EXISTS — safe to re-run.
SCHEMA_SQL = """
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted_at);
CREATE INDEX IF NOT EXISTS idx_memories_canonical ON memories(canonical_type);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed_at);

CREATE TABLE IF NOT EXISTS tags (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  name        TEXT UNIQUE NOT NULL COLLATE NOCASE,
  created_at  DATETIME NOT NULL DEFAULT (datetime('now')),
  use_count   INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_tags_use ON tags(use_count DESC);

CREATE TABLE IF NOT EXISTS memory_tags (
  memory_id  INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  tag_id     INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
  PRIMARY KEY (memory_id, tag_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_tags_tag ON memory_tags(tag_id);

CREATE TRIGGER IF NOT EXISTS memory_tags_ai AFTER INSERT ON memory_tags BEGIN
  UPDATE tags SET use_count = use_count + 1 WHERE id = NEW.tag_id;
END;

CREATE TRIGGER IF NOT EXISTS memory_tags_ad AFTER DELETE ON memory_tags BEGIN
  UPDATE tags SET use_count = use_count - 1 WHERE id = OLD.tag_id;
END;

CREATE TABLE IF NOT EXISTS type_aliases (
  alias       TEXT PRIMARY KEY,
  canonical   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_meta (
  memory_id     INTEGER PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
  status        TEXT CHECK (status IN ('pending', 'doing', 'done', 'waiting', 'cancelled')),
  priority      TEXT CHECK (priority IN ('p1', 'p2', 'p3', 'p4')),
  energy        TEXT CHECK (energy IN ('high', 'medium', 'low')),
  points        INTEGER,
  due_at        DATETIME,
  completed_at  DATETIME,
  external_ref  TEXT,
  parent_uid    TEXT
);
CREATE INDEX IF NOT EXISTS idx_task_status ON task_meta(status) WHERE status IN ('pending', 'doing');
CREATE INDEX IF NOT EXISTS idx_task_due ON task_meta(due_at) WHERE status IN ('pending', 'doing');

CREATE TABLE IF NOT EXISTS memory_links (
  src_id      INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  dst_id      INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  kind        TEXT NOT NULL CHECK (kind IN (
                'cites', 'caused_by', 'fixed_by', 'superseded_by',
                'blocks', 'related_to', 'duplicate_of'
              )),
  created_at  DATETIME NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_links_dst ON memory_links(dst_id, kind);

CREATE TABLE IF NOT EXISTS memory_versions (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id    INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  version      INTEGER NOT NULL,
  title        TEXT,
  content      TEXT,
  changed_at   DATETIME NOT NULL DEFAULT (datetime('now')),
  changed_by   TEXT,
  diff_summary TEXT,
  UNIQUE(memory_id, version)
);
CREATE INDEX IF NOT EXISTS idx_versions_memory ON memory_versions(memory_id);

CREATE TRIGGER IF NOT EXISTS memories_version_on_update
  BEFORE UPDATE ON memories
  WHEN OLD.content IS NOT NEW.content OR OLD.title IS NOT NEW.title
BEGIN
  INSERT INTO memory_versions(memory_id, version, title, content, changed_by)
  VALUES (OLD.id, OLD.version, OLD.title, OLD.content, 'auto');
  UPDATE memories SET version = OLD.version + 1 WHERE id = OLD.id;
END;

CREATE TABLE IF NOT EXISTS alterations (
  id          INTEGER PRIMARY KEY,
  memory_uid  TEXT NOT NULL,
  ts          TEXT NOT NULL,
  kind        TEXT NOT NULL,
  delta       TEXT,
  reason      TEXT
);
CREATE INDEX IF NOT EXISTS idx_alterations_uid ON alterations(memory_uid);

-- v4: which memories were recently surfaced into an agent's context.
-- `brain reconcile` uses this as a re-extraction guard: a "new" fact that
-- closely matches a recently-recalled memory is almost always the agent
-- re-saving its own context (the Mem0 808-duplicate feedback loop).
CREATE TABLE IF NOT EXISTS recall_log (
  id           INTEGER PRIMARY KEY,
  memory_id    INTEGER NOT NULL,
  recalled_at  DATETIME NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_recall_log_time ON recall_log(recalled_at);
CREATE INDEX IF NOT EXISTS idx_recall_log_memory ON recall_log(memory_id);

-- v5: declared identity. Partial-unique so at most one LIVE memory can hold a
-- given identity; invalidated/deleted predecessors keep theirs for history.
CREATE UNIQUE INDEX IF NOT EXISTS idx_memories_identity
  ON memories(identity_key)
  WHERE identity_key IS NOT NULL AND deleted_at IS NULL AND invalid_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memories_anchor ON memories(anchor) WHERE anchor IS NOT NULL;

-- v7: artifact graph. A memory cannot hold a 126MB db dump, so it holds a
-- POINTER plus the file's state at the moment the fact was recorded. `brain
-- artifact check` re-stats them, so a memory can tell you not just where the
-- file is but whether it still says what it said. 41% of the filesystem paths
-- referenced by the corpus on 2026-07-27 no longer existed — dangling
-- pointers that nothing was tracking.
CREATE TABLE IF NOT EXISTS artifacts (
  id           INTEGER PRIMARY KEY,
  path         TEXT NOT NULL UNIQUE,   -- as written (may contain ~)
  real_path    TEXT NOT NULL,          -- expanduser'd, for stat()
  kind         TEXT,                   -- file | dir
  size         INTEGER,
  sha256       TEXT,                   -- NULL when over the hash cap
  mtime        TEXT,
  first_seen   TEXT NOT NULL DEFAULT (datetime('now')),
  last_checked TEXT,
  missing_at   TEXT,                   -- soft, mirrors invalid_at
  changed_at   TEXT                    -- last time size/hash moved
);
CREATE INDEX IF NOT EXISTS idx_artifacts_missing ON artifacts(missing_at);

CREATE TABLE IF NOT EXISTS memory_artifacts (
  memory_id   INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
  artifact_id INTEGER NOT NULL REFERENCES artifacts(id) ON DELETE CASCADE,
  PRIMARY KEY (memory_id, artifact_id)
);
CREATE INDEX IF NOT EXISTS idx_memart_artifact ON memory_artifacts(artifact_id);

-- v5: usefulness signal. recall_log records that a memory was SHOWN; this
-- records whether it actually helped. Feeds a small capped term in ranking.
CREATE TABLE IF NOT EXISTS memory_feedback (
  id        INTEGER PRIMARY KEY,
  memory_id INTEGER NOT NULL,
  signal    INTEGER NOT NULL,
  note      TEXT,
  ts        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_feedback_memory ON memory_feedback(memory_id);

-- v5: pinned core blocks — a few short labelled values injected into EVERY
-- `brain context` regardless of the query. char_limit is enforced so this can
-- never become the unbounded-injection problem it exists to prevent.
CREATE TABLE IF NOT EXISTS blocks (
  label      TEXT PRIMARY KEY,
  value      TEXT NOT NULL,
  char_limit INTEGER NOT NULL DEFAULT 400,
  position   INTEGER NOT NULL DEFAULT 100,
  updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

TYPE_ALIASES = [
    ("learning", "learning"), ("lesson", "learning"), ("insight", "learning"),
    ("decision", "decision"), ("choice", "decision"),
    ("bug", "bug"), ("fix", "bug"), ("issue", "bug"),
    ("snippet", "snippet"), ("code", "snippet"), ("pattern", "snippet"), ("recipe", "snippet"),
    ("note", "note"), ("reference", "note"), ("checklist", "note"), ("merge-snapshot", "note"),
    ("task", "task"), ("todo", "task"),
    ("person", "person"), ("contact", "person"),
    ("project", "project"),
]


def open_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA trusted_schema = ON")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def existing_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def add_columns(conn: sqlite3.Connection) -> int:
    have = existing_columns(conn, "memories")
    added = 0
    for col, typedef in NEW_COLUMNS:
        if col in have:
            continue
        conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {typedef}")
        added += 1
    conn.commit()
    return added


def apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    conn.commit()


def insert_aliases(conn: sqlite3.Connection) -> int:
    n = 0
    for alias, canonical in TYPE_ALIASES:
        cur = conn.execute(
            "INSERT OR IGNORE INTO type_aliases(alias, canonical) VALUES (?, ?)",
            (alias, canonical),
        )
        n += cur.rowcount
    conn.commit()
    return n


def backfill_content_hash(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT id, COALESCE(title, ''), COALESCE(content, '') FROM memories WHERE content_hash IS NULL"
    ).fetchall()
    for memory_id, title, content in rows:
        h = hashlib.sha256(f"{title}\n{content}".encode()).hexdigest()[:16]
        conn.execute("UPDATE memories SET content_hash = ? WHERE id = ?", (h, memory_id))
    conn.commit()
    return len(rows)


def backfill_canonical_type(conn: sqlite3.Connection) -> int:
    aliases = dict(conn.execute("SELECT alias, canonical FROM type_aliases").fetchall())
    rows = conn.execute("SELECT id, type FROM memories WHERE canonical_type IS NULL").fetchall()
    for memory_id, raw_type in rows:
        canonical = aliases.get((raw_type or "").lower(), "note")
        conn.execute("UPDATE memories SET canonical_type = ? WHERE id = ?", (canonical, memory_id))
    conn.commit()
    return len(rows)


def backfill_tags(conn: sqlite3.Connection) -> tuple[int, int]:
    rows = conn.execute(
        "SELECT id, tags FROM memories WHERE tags IS NOT NULL AND tags != ''"
    ).fetchall()
    memories_done = 0
    links = 0
    for memory_id, tags_csv in rows:
        for raw in tags_csv.split(","):
            tag = raw.strip().lower()
            if not tag:
                continue
            conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag,))
            tag_id = conn.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()[0]
            try:
                conn.execute(
                    "INSERT INTO memory_tags(memory_id, tag_id) VALUES (?, ?)",
                    (memory_id, tag_id),
                )
                links += 1
            except sqlite3.IntegrityError:
                pass
        memories_done += 1
    conn.commit()
    return memories_done, links


def backfill_task_meta(conn: sqlite3.Connection) -> int:
    rows = conn.execute("""
        SELECT id, status, priority, energy, points, completed_at
        FROM memories
        WHERE type = 'task'
          AND id NOT IN (SELECT memory_id FROM task_meta)
    """).fetchall()
    for memory_id, status, priority, energy, points, completed_at in rows:
        # CHECK constraints fail on out-of-vocab values; clamp to NULL so backfill never crashes.
        status = status if status in ("pending", "doing", "done", "waiting", "cancelled") else None
        priority = priority if priority in ("p1", "p2", "p3", "p4") else None
        energy = energy if energy in ("high", "medium", "low") else None
        conn.execute("""
            INSERT INTO task_meta(memory_id, status, priority, energy, points, completed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (memory_id, status, priority, energy, points, completed_at))
    conn.commit()
    return len(rows)


def setup_vec_extension(conn: sqlite3.Connection) -> bool:
    try:
        import sqlite_vec
    except ImportError:
        print("    ⚠ sqlite-vec not installed (pip install sqlite-vec). Skipping vector table.")
        return False
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        # Legacy single-vector table (kept for rollback; no longer written to
        # once memory_chunks exists).
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                memory_id INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            )
        """)
        # Chunked embeddings, cosine distance. One row per ~500-char chunk;
        # rowid = memory_id * 64 + chunk_index, memory_id is a filterable
        # metadata column returned alongside KNN distances.
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_chunks USING vec0(
                memory_id INTEGER,
                embedding FLOAT[384] distance_metric=cosine
            )
        """)
        conn.commit()
        return True
    except Exception as e:
        print(f"    ⚠ vec0 setup failed: {e}")
        return False


def fts_needs_rebuild(conn: sqlite3.Connection) -> bool:
    """True when memories_fts is missing, not porter-tokenized, or predates the
    v6 `anchor` column.

    v6: anchor was added to `memories` in v5 but indexed NOWHERE — not in FTS
    (title/content/tags/project/type) and not in the embedded chunk text
    (title + content). So it only affected `context` grouping and could not
    help retrieval at all. Indexing it is the whole point: an anchor is the
    short entity token that disambiguates a dense cluster of sibling memories
    when you cannot recall the distinctive proper noun.

    `abstract` is deliberately NOT indexed: it is a restatement of content, so
    indexing it would double term frequencies for summarized memories and skew
    BM25 toward whichever memories happen to have an abstract.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'memories_fts' AND type = 'table'"
    ).fetchone()
    sql = (row[0] or "") if row else ""
    return not (row and "porter" in sql and "anchor" in sql)


def rebuild_fts_porter(conn: sqlite3.Connection) -> bool:
    """Recreate memories_fts with porter stemming (migration/migrations match).

    External-content FTS — dropping and rebuilding loses nothing; content
    lives in memories. Triggers are recreated identically. Idempotent: skips
    when the table already uses porter.
    """
    if not fts_needs_rebuild(conn):
        return False
    conn.executescript("""
        DROP TRIGGER IF EXISTS memories_ai;
        DROP TRIGGER IF EXISTS memories_au;
        DROP TRIGGER IF EXISTS memories_ad;
        DROP TABLE IF EXISTS memories_fts;
        CREATE VIRTUAL TABLE memories_fts USING fts5(
          title, content, tags, project, type, anchor,
          content=memories, content_rowid=rowid,
          tokenize='porter unicode61'
        );
        CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
          INSERT INTO memories_fts(rowid, title, content, tags, project, type, anchor)
          VALUES (new.rowid, new.title, new.content, new.tags, new.project, new.type, new.anchor);
        END;
        CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, title, content, tags, project, type, anchor)
          VALUES ('delete', old.rowid, old.title, old.content, old.tags, old.project, old.type, old.anchor);
          INSERT INTO memories_fts(rowid, title, content, tags, project, type, anchor)
          VALUES (new.rowid, new.title, new.content, new.tags, new.project, new.type, new.anchor);
        END;
        CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, title, content, tags, project, type, anchor)
          VALUES ('delete', old.rowid, old.title, old.content, old.tags, old.project, old.type, old.anchor);
        END;
        INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');
    """)
    conn.commit()
    return True


def mark_schema_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO stats(key, value, updated_at) VALUES "
        f"('brain_schema_version', '{SCHEMA_VERSION}', datetime('now'))"
    )
    conn.commit()


def read_schema_version(conn: sqlite3.Connection) -> int | None:
    """Current brain_schema_version as int, or None (pre-v3 / unparseable)."""
    try:
        row = conn.execute(
            "SELECT value FROM stats WHERE key = 'brain_schema_version'"
        ).fetchone()
    except sqlite3.OperationalError:
        return None  # no stats table yet — pre-v3 DB
    if not row:
        return None
    try:
        return int(str(row[0]).strip())
    except (TypeError, ValueError):
        return None


def build_plan(conn: sqlite3.Connection) -> dict:
    """Inspect the DB without writing; returns what migration would do."""
    have_cols = existing_columns(conn, "memories")
    have_tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    return {
        "columns_to_add": [c for c, _ in NEW_COLUMNS if c not in have_cols],
        "tables_to_create": [t for t in SCHEMA_TABLES + VEC_TABLES if t not in have_tables],
        "fts_rebuild": fts_needs_rebuild(conn),
    }


def backup_db(conn: sqlite3.Connection) -> Path:
    """Snapshot the DB next to itself before destructive work.

    Uses the sqlite backup API (not a file copy) so pending WAL pages are
    included and the snapshot is consistent even with the connection open.
    """
    dest = DB.with_name(f"{DB.name}.bak-migrate-{time.strftime('%Y%m%d-%H%M%S')}")
    out = sqlite3.connect(dest)
    try:
        conn.backup(out)
    finally:
        out.close()
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="brain v3 migration runner (idempotent)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would run and exit without writing")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the pre-destructive-step DB backup")
    args = ap.parse_args()

    if not DB.exists():
        print(f"✗ {DB} does not exist. Aborting.", file=sys.stderr)
        return 1

    conn = open_db()

    # Forward-compatibility guard: never run an old migrator on a newer DB.
    db_version = read_schema_version(conn)
    if db_version is not None and db_version > SCHEMA_VERSION:
        print(
            f"✗ {DB} reports schema version {db_version}, but this script only "
            f"produces version {SCHEMA_VERSION}. Refusing to migrate a "
            f"forward-incompatible DB — upgrade brain-cli instead.",
            file=sys.stderr,
        )
        conn.close()
        return 1

    plan = build_plan(conn)
    will_backup = plan["fts_rebuild"] and not args.no_backup

    if args.dry_run:
        print(f"→ dry-run for {DB} (schema version: {db_version or 'unset'})")
        print(f"  • columns to add: {', '.join(plan['columns_to_add']) or '(none)'}")
        print(f"  • tables to create: {', '.join(plan['tables_to_create']) or '(none)'}")
        print(f"  • FTS porter rebuild: {'yes (destructive drop/rebuild)' if plan['fts_rebuild'] else 'no — already porter'}")
        print(f"  • backup: {'yes — .bak-migrate-<timestamp>' if will_backup else 'no' + (' (--no-backup)' if args.no_backup and plan['fts_rebuild'] else ' — nothing destructive')}")
        print("  (no changes written)")
        conn.close()
        return 0

    print(f"→ migrating {DB}")

    if will_backup:
        dest = backup_db(conn)
        print(f"  • backup → {dest}")

    print("  • adding new columns to memories")
    n = add_columns(conn)
    print(f"    {n} column(s) added")

    print("  • applying tables, indexes, triggers")
    apply_schema(conn)

    print("  • inserting type aliases")
    n = insert_aliases(conn)
    print(f"    {n} new alias(es)")

    print("  • backfilling content_hash")
    n = backfill_content_hash(conn)
    print(f"    {n} memories hashed")

    print("  • backfilling canonical_type")
    n = backfill_canonical_type(conn)
    print(f"    {n} memories canonicalized")

    print("  • backfilling tags table from CSV")
    memories_done, links = backfill_tags(conn)
    print(f"    {memories_done} memories → {links} tag links")

    print("  • backfilling task_meta")
    n = backfill_task_meta(conn)
    print(f"    {n} tasks materialized")

    print("  • setting up sqlite-vec memory_vectors + memory_chunks")
    if setup_vec_extension(conn):
        print("    ✓ ready (run `brain reindex` to populate)")

    print("  • rebuilding memories_fts with porter stemming")
    if rebuild_fts_porter(conn):
        print("    ✓ rebuilt (porter unicode61)")
    else:
        print("    already porter — skipped")

    mark_schema_version(conn)
    print(f"\n✓ schema v{SCHEMA_VERSION} applied")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
