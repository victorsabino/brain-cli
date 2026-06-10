#!/usr/bin/env python3
"""brain v3 migration runner. Idempotent. Safe to re-run."""

from __future__ import annotations
import hashlib
import os
import sqlite3
import sys
from pathlib import Path

DB = Path(os.environ.get("BRAIN_DB") or (Path.home() / "brain.db"))
HERE = Path(__file__).resolve().parent

# All ALTER TABLE statements run individually with duplicate-tolerance.
NEW_COLUMNS = [
    ("content_hash", "TEXT"),
    ("deleted_at", "DATETIME"),
    ("last_accessed_at", "DATETIME"),
    ("access_count", "INTEGER NOT NULL DEFAULT 0"),
    ("version", "INTEGER NOT NULL DEFAULT 1"),
    ("superseded_by", "TEXT"),
    ("canonical_type", "TEXT"),
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


def rebuild_fts_porter(conn: sqlite3.Connection) -> bool:
    """Recreate memories_fts with porter stemming (migration/migrations match).

    External-content FTS — dropping and rebuilding loses nothing; content
    lives in memories. Triggers are recreated identically. Idempotent: skips
    when the table already uses porter.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'memories_fts' AND type = 'table'"
    ).fetchone()
    if row and "porter" in (row[0] or ""):
        return False
    conn.executescript("""
        DROP TRIGGER IF EXISTS memories_ai;
        DROP TRIGGER IF EXISTS memories_au;
        DROP TRIGGER IF EXISTS memories_ad;
        DROP TABLE IF EXISTS memories_fts;
        CREATE VIRTUAL TABLE memories_fts USING fts5(
          title, content, tags, project, type,
          content=memories, content_rowid=rowid,
          tokenize='porter unicode61'
        );
        CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
          INSERT INTO memories_fts(rowid, title, content, tags, project, type)
          VALUES (new.rowid, new.title, new.content, new.tags, new.project, new.type);
        END;
        CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, title, content, tags, project, type)
          VALUES ('delete', old.rowid, old.title, old.content, old.tags, old.project, old.type);
          INSERT INTO memories_fts(rowid, title, content, tags, project, type)
          VALUES (new.rowid, new.title, new.content, new.tags, new.project, new.type);
        END;
        CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
          INSERT INTO memories_fts(memories_fts, rowid, title, content, tags, project, type)
          VALUES ('delete', old.rowid, old.title, old.content, old.tags, old.project, old.type);
        END;
        INSERT INTO memories_fts(memories_fts) VALUES ('rebuild');
    """)
    conn.commit()
    return True


def mark_schema_version(conn: sqlite3.Connection) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO stats(key, value, updated_at) VALUES ('brain_schema_version', '3', datetime('now'))"
    )
    conn.commit()


def main() -> int:
    if not DB.exists():
        print(f"✗ {DB} does not exist. Aborting.", file=sys.stderr)
        return 1

    print(f"→ migrating {DB}")
    conn = open_db()

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
    print("\n✓ schema v3 applied")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
