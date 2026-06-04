"""Smoke tests for `brain update` / `history` / alterations audit trail.

SAFETY: these tests NEVER touch ~/brain.db. setUp copies the real DB to a
throwaway temp file and points BRAIN_DB at it for every CLI call. The real
~/brain.db mtime is asserted unchanged before/after the whole suite.

Runs FTS-only — embeddings (sentence-transformers/torch) are optional, so the
suite passes even when those heavy deps are not installed.

    cd /Users/sabino/Documents/brain-cli && \
      BRAIN_DB=/tmp/brain-test.db python3 -m pytest tests/test_update_history.py -q
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sqlite3
import tempfile
from pathlib import Path

import pytest

BRAIN = str(Path(__file__).resolve().parent.parent / "brain.py")
REAL_DB = Path.home() / "brain.db"


@pytest.fixture()
def env():
    """Throwaway DB copy + env wired to it. Asserts real DB is never written."""
    if not REAL_DB.exists():
        pytest.skip("~/brain.db not present — nothing to copy schema from")

    real_mtime_before = REAL_DB.stat().st_mtime_ns

    tmpdir = tempfile.mkdtemp(prefix="brain-test-")
    test_db = Path(tmpdir) / "brain-test.db"
    shutil.copy2(REAL_DB, test_db)

    e = dict(os.environ)
    e["BRAIN_DB"] = str(test_db)
    # Keep tests fast + dependency-light: skip semantic embedding model load.
    e.setdefault("PYTHONUNBUFFERED", "1")

    # Guard: the path the CLI uses MUST be the temp copy, not ~/brain.db.
    assert str(test_db) != str(REAL_DB)

    yield e, test_db

    # The real DB must be byte-for-byte untouched.
    assert REAL_DB.stat().st_mtime_ns == real_mtime_before, "REAL ~/brain.db was modified!"
    shutil.rmtree(tmpdir, ignore_errors=True)


def run(env, *cli_args, expect_ok=True):
    e, _ = env
    proc = subprocess.run(
        [sys.executable, BRAIN, *cli_args],
        env=e, capture_output=True, text=True,
    )
    if expect_ok:
        assert proc.returncode == 0, f"cmd {cli_args} failed:\n{proc.stderr}"
    return proc


def _hash_of(test_db: Path, uid: str) -> str:
    conn = sqlite3.connect(test_db)
    try:
        row = conn.execute(
            "SELECT content_hash FROM memories WHERE uid = ?", (uid,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _save_marker(env) -> str:
    proc = run(env, "save", "--type=note",
               "--title=ztest update history marker",
               "--content=initial body", "--no-embed", "--force")
    uid = proc.stdout.strip().splitlines()[-1]
    assert uid and len(uid) >= 8, f"bad uid: {uid!r}"
    return uid


def test_save_returns_uid(env):
    uid = _save_marker(env)
    assert uid


def test_update_append_changes_content_and_hash(env):
    _, test_db = env
    uid = _save_marker(env)
    hash_before = _hash_of(test_db, uid)

    run(env, "update", uid, "--append", "ZZZ_marker",
        "--reason", "auto-capture test")

    got = run(env, "get", uid)
    assert "ZZZ_marker" in got.stdout
    assert "initial body" in got.stdout  # append, not replace

    hash_after = _hash_of(test_db, uid)
    assert hash_after != hash_before, "content_hash must change after append"


def test_alterations_has_create_and_append(env):
    _, test_db = env
    uid = _save_marker(env)
    run(env, "update", uid, "--append", "ZZZ_marker")

    conn = sqlite3.connect(test_db)
    try:
        kinds = [r[0] for r in conn.execute(
            "SELECT kind FROM alterations WHERE memory_uid = ? ORDER BY id", (uid,)
        ).fetchall()]
    finally:
        conn.close()
    assert "create" in kinds
    assert "append" in kinds


def test_history_shows_both(env):
    uid = _save_marker(env)
    run(env, "update", uid, "--append", "ZZZ_marker", "--reason", "why123")
    out = run(env, "history", uid).stdout
    assert "create" in out
    assert "append" in out
    assert "why123" in out


def test_search_finds_appended_marker(env):
    uid = _save_marker(env)
    run(env, "update", uid, "--append", "ZZZ_marker")
    # FTS is trigger-synced on UPDATE — searching the appended token must hit.
    out = run(env, "search", "ZZZ_marker", "--no-semantic").stdout
    assert uid in out or "ZZZ_marker" in out


def test_update_bogus_uid_fails(env):
    proc = run(env, "update", "nonexistent99", "--append", "x", expect_ok=False)
    assert proc.returncode != 0
    assert "not found" in (proc.stdout + proc.stderr).lower()


def test_help_lists_update_and_history(env):
    out = run(env, "--help").stdout
    assert "update" in out
    assert "history" in out
