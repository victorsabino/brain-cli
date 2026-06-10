"""`brain save`: uid contract, dedup, tag normalization, task_meta."""
from __future__ import annotations

import re
import sqlite3

from conftest import run_brain

UID_RE = re.compile(r"^[a-z2-7]{12}$")  # 12-char lowercase base32


def _query(db, sql, *params):
    conn = sqlite3.connect(db)
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def test_save_prints_uid(db):
    r = run_brain(db, "save", "--type=note", "--title=alpha widget calibration",
                  "--content=body text", "--no-embed")
    assert r.returncode == 0, r.stderr
    uid = r.stdout.strip().splitlines()[-1]
    assert UID_RE.match(uid), f"bad uid: {uid!r}"


def test_save_dedup_exits_2_with_existing_uid(db):
    r1 = run_brain(db, "save", "--type=note", "--title=dup title", "--content=dup body",
                   "--no-embed")
    uid1 = r1.stdout.strip()
    r2 = run_brain(db, "save", "--type=note", "--title=dup title", "--content=dup body",
                   "--no-embed")
    assert r2.returncode == 2
    assert "DUPLICATE" in r2.stderr
    assert r2.stdout.strip() == uid1  # prints the EXISTING uid for merge flows


def test_save_force_bypasses_dedup(db):
    r1 = run_brain(db, "save", "--type=note", "--title=dup title", "--content=dup body",
                   "--no-embed")
    r2 = run_brain(db, "save", "--type=note", "--title=dup title", "--content=dup body",
                   "--no-embed", "--force")
    assert r2.returncode == 0, r2.stderr
    assert r2.stdout.strip() != r1.stdout.strip()


def test_save_tags_normalized_lowercase(db):
    r = run_brain(db, "save", "--type=note", "--title=tagged memory",
                  "--tags=Alpha, BETA", "--tags=gamma", "--no-embed")
    assert r.returncode == 0, r.stderr
    names = {row[0] for row in _query(db, "SELECT name FROM tags")}
    assert {"alpha", "beta", "gamma"} <= names
    links = _query(db, """
        SELECT t.name FROM memory_tags mt JOIN tags t ON t.id = mt.tag_id
        JOIN memories m ON m.id = mt.memory_id WHERE m.uid = ?
    """, r.stdout.strip())
    assert {row[0] for row in links} == {"alpha", "beta", "gamma"}


def test_save_task_creates_task_meta_row(db):
    r = run_brain(db, "save", "--type=task", "--title=rotate the demo secrets",
                  "--status=doing", "--priority=p2", "--energy=low", "--points=30",
                  "--no-embed")
    assert r.returncode == 0, r.stderr
    rows = _query(db, """
        SELECT tm.status, tm.priority, tm.energy, tm.points
        FROM task_meta tm JOIN memories m ON m.id = tm.memory_id WHERE m.uid = ?
    """, r.stdout.strip())
    assert rows == [("doing", "p2", "low", 30)]


def test_save_type_alias_canonicalized(db):
    r = run_brain(db, "save", "--type=insight", "--title=alias check", "--no-embed")
    assert r.returncode == 0, r.stderr
    rows = _query(db, "SELECT type, canonical_type FROM memories WHERE uid = ?",
                  r.stdout.strip())
    assert rows == [("insight", "learning")]


def test_save_unknown_type_maps_to_note(db):
    r = run_brain(db, "save", "--type=BOGUS", "--title=unknown type check", "--no-embed")
    assert r.returncode == 0, r.stderr
    rows = _query(db, "SELECT canonical_type FROM memories WHERE uid = ?", r.stdout.strip())
    assert rows == [("note",)]


def test_save_empty_title_rejected(db):
    r = run_brain(db, "save", "--type=note", "--title=   ", "--no-embed")
    assert r.returncode == 1
    assert "title" in r.stderr.lower()
