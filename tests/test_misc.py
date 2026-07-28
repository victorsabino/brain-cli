"""stats/recent/tags/get smoke on the synthetic fixture (ports old smoke tests)."""
from __future__ import annotations

import json

from conftest import run_brain


def test_stats_reports_schema_and_counts(fixture_db):
    db, _ = fixture_db
    r = run_brain(db, "stats")
    assert r.returncode == 0, r.stderr
    assert "brain v7" in r.stdout
    assert "active:" in r.stdout
    assert "by type:" in r.stdout


def test_recent_lists_rows(fixture_db):
    db, _ = fixture_db
    r = run_brain(db, "recent", "3")
    assert r.returncode == 0, r.stderr
    assert len(r.stdout.strip().splitlines()) == 3


def test_tags_lists_use_counts(fixture_db):
    db, _ = fixture_db
    r = run_brain(db, "tags", "--limit=5")
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip(), "fixture has tags — listing must not be empty"


def test_get_json_roundtrip(fixture_db):
    db, _ = fixture_db
    listing = json.loads(run_brain(db, "search", "", "--type=person", "--json").stdout)
    uid = listing[0]["uid"]
    r = run_brain(db, "get", uid, "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["uid"] == uid
    assert "content_hash" not in d  # stripped from JSON output


def test_get_unknown_uid_fails(fixture_db):
    db, _ = fixture_db
    r = run_brain(db, "get", "zzzzzzzzzzzz")
    assert r.returncode == 1
    assert "not found" in r.stderr
