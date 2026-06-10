"""scripts/migrate.py: dry-run is read-only, version guard, FTS-rebuild backup."""
from __future__ import annotations

import hashlib
import sqlite3

from conftest import create_base_db, run_migrate

NON_PORTER_FTS = """
CREATE VIRTUAL TABLE memories_fts USING fts5(
  title, content, tags, project, type,
  content=memories, content_rowid=rowid
);
"""


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fts_sql(db) -> str:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'memories_fts' AND type = 'table'"
        ).fetchone()
        return (row[0] or "") if row else ""
    finally:
        conn.close()


def test_dry_run_writes_nothing(tmp_path):
    db = tmp_path / "brain.db"
    create_base_db(db)  # pre-v3 → plan is non-empty, real run WOULD write
    before = _sha(db)
    r = run_migrate(db, "--dry-run")
    assert r.returncode == 0, r.stderr
    assert "(no changes written)" in r.stdout
    assert "columns to add" in r.stdout
    assert _sha(db) == before, "--dry-run modified the DB file"


def test_refuses_newer_schema_version(db):
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT OR REPLACE INTO stats(key, value) VALUES ('brain_schema_version', '99')"
    )
    conn.commit()
    conn.close()
    before = _sha(db)
    r = run_migrate(db)
    assert r.returncode == 1
    assert "Refusing" in r.stderr
    assert _sha(db) == before, "refused migration must not write"


def test_backup_created_on_fts_porter_rebuild(tmp_path):
    db = tmp_path / "brain.db"
    create_base_db(db)
    conn = sqlite3.connect(db)
    conn.executescript(NON_PORTER_FTS)  # forces the destructive rebuild path
    conn.commit()
    conn.close()
    assert "porter" not in _fts_sql(db)

    r = run_migrate(db)
    assert r.returncode == 0, r.stderr
    backups = list(tmp_path.glob("brain.db.bak-migrate-*"))
    assert len(backups) == 1, "FTS rebuild must snapshot the DB first"
    assert "porter" in _fts_sql(db)

    # Idempotent re-run: already porter → no rebuild, no second backup.
    r = run_migrate(db)
    assert r.returncode == 0, r.stderr
    assert "already porter" in r.stdout
    assert len(list(tmp_path.glob("brain.db.bak-migrate-*"))) == 1


def test_no_backup_flag(tmp_path):
    db = tmp_path / "brain.db"
    create_base_db(db)
    conn = sqlite3.connect(db)
    conn.executescript(NON_PORTER_FTS)
    conn.commit()
    conn.close()
    r = run_migrate(db, "--no-backup")
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.glob("brain.db.bak-migrate-*")) == []
