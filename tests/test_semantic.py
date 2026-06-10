"""Model-dependent tests — skipped in CI (no sentence-transformers by design).

Each subprocess pays the lazy model load (~10s), so keep this file minimal:
one save-path test, one doctor-repair test.
"""
from __future__ import annotations

import sqlite3

from conftest import requires_model, run_brain

pytestmark = requires_model


def _chunk_rows(db, memory_id: int) -> list[int]:
    import sqlite_vec
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    try:
        return [r[0] for r in conn.execute(
            "SELECT rowid FROM memory_chunks WHERE memory_id = ?", (memory_id,))]
    finally:
        conn.close()


def test_save_with_embedding_writes_chunks_in_rowid_range(db):
    r = run_brain(db, "save", "--type=note", "--title=embedded memory",
                  "--content=" + "\n\n".join(f"paragraph {i} about harbor kiosks" * 8
                                             for i in range(4)))
    assert r.returncode == 0, r.stderr
    uid = r.stdout.strip()
    conn = sqlite3.connect(db)
    memory_id = conn.execute("SELECT id FROM memories WHERE uid = ?", (uid,)).fetchone()[0]
    conn.close()
    rowids = _chunk_rows(db, memory_id)
    assert rowids, "save without --no-embed must write chunk vectors"
    base = memory_id * 64
    assert all(base <= rid < base + 64 for rid in rowids)


def test_doctor_fix_reembeds_missing(db):
    run_brain(db, "save", "--type=note", "--title=unembedded memory", "--no-embed")
    r = run_brain(db, "doctor", "--fix")
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "re-embedded 1/1" in r.stdout
    r = run_brain(db, "doctor")
    assert r.returncode == 0
    assert "healthy" in r.stdout
