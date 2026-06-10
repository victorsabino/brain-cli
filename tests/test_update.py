"""`brain update` / `history`: dated bullets, FTS trigger sync, audit trail,
version snapshots. Ports the old live-DB smoke assertions onto hermetic DBs."""
from __future__ import annotations

import re
import sqlite3

from conftest import run_brain


def _query(db, sql, *params):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _save_marker(db) -> str:
    r = run_brain(db, "save", "--type=note", "--title=update history marker",
                  "--content=initial body", "--no-embed")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip().splitlines()[-1]


def test_append_adds_dated_bullet_and_keeps_original(db):
    uid = _save_marker(db)
    r = run_brain(db, "update", uid, "--append", "galvanizing the kiosk fleet")
    assert r.returncode == 0, r.stderr
    content = _query(db, "SELECT content FROM memories WHERE uid = ?", uid)[0][0]
    assert "initial body" in content  # append, not replace
    assert re.search(r"\n\n- \[\d{4}-\d{2}-\d{2}\] galvanizing the kiosk fleet$", content)


def test_append_changes_content_hash(db):
    uid = _save_marker(db)
    before = _query(db, "SELECT content_hash FROM memories WHERE uid = ?", uid)[0][0]
    run_brain(db, "update", uid, "--append", "delta")
    after = _query(db, "SELECT content_hash FROM memories WHERE uid = ?", uid)[0][0]
    assert after != before


def test_fts_trigger_sync_porter_stemmed_match(db):
    """The AFTER UPDATE trigger must re-sync memories_fts, and porter stemming
    means a STEMMED variant of the appended word ('galvanized' vs the stored
    'galvanizing') still hits."""
    uid = _save_marker(db)
    run_brain(db, "update", uid, "--append", "galvanizing the kiosk fleet")
    r = run_brain(db, "search", "galvanized", "--no-semantic")
    assert r.returncode == 0, r.stderr
    assert uid in r.stdout


def test_alterations_history_create_then_append(db):
    uid = _save_marker(db)
    run_brain(db, "update", uid, "--append", "more detail", "--reason", "why123")
    kinds = [row[0] for row in _query(
        db, "SELECT kind FROM alterations WHERE memory_uid = ? ORDER BY id", uid)]
    assert kinds == ["create", "append"]
    out = run_brain(db, "history", uid).stdout
    assert "create" in out and "append" in out and "why123" in out


def test_update_snapshots_previous_version(db):
    """BEFORE UPDATE trigger writes the old content to memory_versions and
    bumps memories.version."""
    uid = _save_marker(db)
    run_brain(db, "update", uid, "--replace", "entirely new body")
    rows = _query(db, """
        SELECT v.content, m.version FROM memory_versions v
        JOIN memories m ON m.id = v.memory_id WHERE m.uid = ?
    """, uid)
    assert rows == [("initial body", 2)]
    assert _query(db, "SELECT content FROM memories WHERE uid = ?", uid)[0][0] == "entirely new body"


def test_update_bogus_uid_fails(db):
    r = run_brain(db, "update", "nonexistent99", "--append", "x")
    assert r.returncode != 0
    assert "not found" in (r.stdout + r.stderr).lower()


def test_update_requires_append_or_replace(db):
    uid = _save_marker(db)
    r = run_brain(db, "update", uid)
    assert r.returncode != 0


def test_get_bumps_access_count(db):
    uid = _save_marker(db)
    run_brain(db, "get", uid)
    run_brain(db, "get", uid)
    assert _query(db, "SELECT access_count FROM memories WHERE uid = ?", uid)[0][0] == 2


def test_help_lists_update_and_history(db):
    out = run_brain(db, "--help").stdout
    assert "update" in out and "history" in out
