-- brain.db v3 migration — additive, idempotent.
-- Run via `brain migrate` or `python scripts/migrate.py`.
-- NEVER drops anything. Existing memories/stats/orch_*/helix_*/sessions/etc tables untouched.

PRAGMA trusted_schema = ON;
PRAGMA foreign_keys = ON;

BEGIN;

------------------------------------------------------------
-- 1. NEW COLUMNS on memories (additive, NULL-safe)
--    SQLite has no IF NOT EXISTS for ALTER ADD COLUMN, so the
--    Python migrator catches "duplicate column name" and continues.
------------------------------------------------------------

-- Filled by migrator; SHA256(title || '\n' || content) hex first 16 chars.
ALTER TABLE memories ADD COLUMN content_hash TEXT;

-- Soft delete: NULL = active, datetime = deleted at.
ALTER TABLE memories ADD COLUMN deleted_at DATETIME;

-- Recency boost.
ALTER TABLE memories ADD COLUMN last_accessed_at DATETIME;
ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;

-- Versioning (snapshot count).
ALTER TABLE memories ADD COLUMN version INTEGER NOT NULL DEFAULT 1;

-- For supersession links (newer memory replaces older one).
ALTER TABLE memories ADD COLUMN superseded_by TEXT;

-- Canonical type after alias resolution. Original `type` column kept for backwards compat.
ALTER TABLE memories ADD COLUMN canonical_type TEXT;

------------------------------------------------------------
-- 2. INDEXES on new columns
------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash) WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted_at);
CREATE INDEX IF NOT EXISTS idx_memories_canonical ON memories(canonical_type);
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(last_accessed_at);

------------------------------------------------------------
-- 3. TAGS — relational, replaces CSV TEXT
------------------------------------------------------------

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

-- Trigger: auto-update use_count when memory_tags change.
CREATE TRIGGER IF NOT EXISTS memory_tags_ai AFTER INSERT ON memory_tags BEGIN
  UPDATE tags SET use_count = use_count + 1 WHERE id = NEW.tag_id;
END;
CREATE TRIGGER IF NOT EXISTS memory_tags_ad AFTER DELETE ON memory_tags BEGIN
  UPDATE tags SET use_count = use_count - 1 WHERE id = OLD.tag_id;
END;

------------------------------------------------------------
-- 4. TYPE ALIASES — collapses 12 types into ~7 canonical
------------------------------------------------------------

CREATE TABLE IF NOT EXISTS type_aliases (
  alias       TEXT PRIMARY KEY,
  canonical   TEXT NOT NULL
);

INSERT OR IGNORE INTO type_aliases(alias, canonical) VALUES
  ('learning', 'learning'),
  ('lesson', 'learning'),
  ('insight', 'learning'),
  ('decision', 'decision'),
  ('choice', 'decision'),
  ('bug', 'bug'),
  ('fix', 'bug'),
  ('issue', 'bug'),
  ('snippet', 'snippet'),
  ('code', 'snippet'),
  ('pattern', 'snippet'),
  ('recipe', 'snippet'),
  ('note', 'note'),
  ('reference', 'note'),
  ('checklist', 'note'),
  ('merge-snapshot', 'note'),
  ('task', 'task'),
  ('todo', 'task'),
  ('person', 'person'),
  ('contact', 'person'),
  ('project', 'project');

------------------------------------------------------------
-- 5. TASK META — typed sub-table for task-only fields
--    (status/priority/energy/points/completed_at stay on memories
--     for backwards compat; new task data writes BOTH places.
--     Drop original cols in v4 once everything cuts over.)
------------------------------------------------------------

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

------------------------------------------------------------
-- 6. MEMORY LINKS — typed relations
------------------------------------------------------------

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

------------------------------------------------------------
-- 7. MEMORY VERSIONS — snapshot before edit (light versioning)
------------------------------------------------------------

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

-- Trigger: snapshot on update if title or content changed.
CREATE TRIGGER IF NOT EXISTS memories_version_on_update
  BEFORE UPDATE ON memories
  WHEN OLD.content IS NOT NEW.content OR OLD.title IS NOT NEW.title
BEGIN
  INSERT INTO memory_versions(memory_id, version, title, content, changed_by)
  VALUES (OLD.id, OLD.version, OLD.title, OLD.content, 'auto');
  UPDATE memories SET version = OLD.version + 1 WHERE id = OLD.id;
END;

------------------------------------------------------------
-- 8. SCHEMA VERSION marker
------------------------------------------------------------

INSERT OR REPLACE INTO stats(key, value, updated_at)
VALUES ('brain_schema_version', '3', datetime('now'));

COMMIT;

-- 9. memory_vectors (sqlite-vec) — created separately by Python migrator
--    because it requires SELECT load_extension(...) which can't be in a
--    plain SQL file safely. See scripts/migrate.py.
