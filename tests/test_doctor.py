"""`brain doctor`: clean baseline, orphan detection, --fix repair, exit codes."""
from __future__ import annotations

import sqlite3
import struct

import pytest
from conftest import HAS_ST, requires_vec, run_brain

FAKE_VEC = struct.pack("384f", *([0.1] * 384))


def _vec_conn(db) -> sqlite3.Connection:
    import sqlite_vec
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def test_doctor_clean_on_fresh_db(db):
    r = run_brain(db, "doctor")
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "healthy" in r.stdout


@requires_vec
def test_doctor_detects_and_fixes_orphan_chunks(db):
    # Manufacture orphans: chunk rows whose memory_id points at nothing.
    conn = _vec_conn(db)
    for ghost_id in (901, 902):
        conn.execute(
            "INSERT INTO memory_chunks(rowid, memory_id, embedding) VALUES (?, ?, ?)",
            (ghost_id * 64, ghost_id, FAKE_VEC),
        )
    conn.commit()
    conn.close()

    r = run_brain(db, "doctor")
    assert r.returncode == 1
    assert "orphans" in r.stdout and "2 chunk rows" in r.stdout

    r = run_brain(db, "doctor", "--fix")
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "deleted 2 orphan chunk rows" in r.stdout

    r = run_brain(db, "doctor")
    assert r.returncode == 0
    assert "healthy" in r.stdout


@requires_vec
def test_doctor_detects_missing_vectors(db):
    run_brain(db, "save", "--type=note", "--title=unembedded memory", "--no-embed")
    r = run_brain(db, "doctor")
    assert r.returncode == 1
    assert "no chunk vectors" in r.stdout


@requires_vec
@pytest.mark.skipif(HAS_ST, reason="model installed — repair path covered in test_semantic")
def test_doctor_fix_without_model_reports_unfixed(db):
    """FTS-only environments can't re-embed: --fix must say so and exit 1,
    never silently claim success."""
    run_brain(db, "save", "--type=note", "--title=unembedded memory", "--no-embed")
    r = run_brain(db, "doctor", "--fix")
    assert r.returncode == 1
    assert "cannot re-embed" in r.stdout
