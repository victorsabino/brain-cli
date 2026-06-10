"""Search behavior on the synthetic fixture corpus, FTS-only (no model).

Relevance is gated by scripts/eval.py against the fixture golden set — the
same gate CI runs — rather than hand-asserting fragile rank positions.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys

from conftest import EVAL, run_brain

EXPLAIN_KEYS = {"fts_rank", "sem_rank", "sim", "rrf_fts", "rrf_sem", "rrf",
                "recency_bonus", "access_bonus", "final"}


def test_eval_gate_fts_only(fixture_db):
    """Golden-set recall@5 on the fixture must clear the CI gate (0.9)."""
    db, golden = fixture_db
    r = subprocess.run(
        [sys.executable, str(EVAL), "--golden", str(golden), "--db", str(db),
         "--no-semantic", "--min-recall5", "0.9"],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "BRAIN_DB": str(db)},
    )
    assert r.returncode == 0, f"eval gate failed:\n{r.stdout}\n{r.stderr}"
    assert "recall@5" in r.stdout


def test_fts_rank_order_sane(fixture_db):
    """A distinctive paraphrase puts its memory at rank 1, not just somewhere."""
    db, _ = fixture_db
    r = run_brain(db, "search", "kiosk stuck rebooting after firmware update",
                  "--no-semantic", "--json")
    hits = json.loads(r.stdout)
    assert hits and "reboot loop" in hits[0]["title"]


def test_empty_query_type_listing(fixture_db):
    """`brain search "" --type=task` is the documented task-listing pattern."""
    db, _ = fixture_db
    r = run_brain(db, "search", "", "--type=task", "--json")
    assert r.returncode == 0, r.stderr
    hits = json.loads(r.stdout)
    assert hits, "listing must return the fixture's tasks"
    assert all(h["type"] == "task" for h in hits)


def test_explain_json_shape(fixture_db):
    db, _ = fixture_db
    r = run_brain(db, "search", "kiosk firmware reboot", "--no-semantic",
                  "--json", "--explain")
    hits = json.loads(r.stdout)
    assert hits
    e = hits[0]["explain"]
    assert set(e) == EXPLAIN_KEYS
    assert e["fts_rank"] == 0          # top FTS hit
    assert e["sem_rank"] is None       # --no-semantic → no semantic rank
    assert e["final"] >= e["rrf"] > 0  # additive bonuses never subtract


def test_no_results_empty_json(fixture_db):
    db, _ = fixture_db
    r = run_brain(db, "search", "zzqx9 nonexistent7 tokens3", "--no-semantic", "--json")
    assert r.returncode == 0
    assert json.loads(r.stdout) == []


def test_type_filter_applies_before_limit(db):
    """80 high-bm25 notes share a token with 1 low-bm25 task. If the type
    filter ran AFTER the FTS LIMIT (pool=50), the task could never surface;
    pre-LIMIT filtering must find it."""
    conn = sqlite3.connect(db)
    filler = "ordinary operational paragraph " * 40
    for i in range(80):
        conn.execute(
            "INSERT INTO memories(uid, type, canonical_type, title, content) "
            "VALUES (?, 'note', 'note', ?, 'short body')",
            (f"note{i:08d}", f"warble calibration note {i}"),  # token in 4x-weighted title
        )
    conn.execute(
        "INSERT INTO memories(uid, type, canonical_type, title, content) "
        "VALUES ('task00000001', 'task', 'task', 'background chore', ?)",
        (filler + " warble",),  # token buried once in 1x-weighted content
    )
    conn.commit()
    conn.close()

    r = run_brain(db, "search", "warble", "--type=task", "--no-semantic", "--json")
    hits = json.loads(r.stdout)
    assert [h["uid"] for h in hits] == ["task00000001"]

    # Sanity: unfiltered, the task is nowhere near the top.
    r = run_brain(db, "search", "warble", "--no-semantic", "--json")
    assert "task00000001" not in [h["uid"] for h in json.loads(r.stdout)]
