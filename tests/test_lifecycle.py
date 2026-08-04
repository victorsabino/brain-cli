"""v4 memory-lifecycle features: invalidate, reconcile, context, consolidate.

All FTS-only (--no-semantic / --no-embed) so they run in CI without the model.
"""
from __future__ import annotations

import json

from conftest import run_brain


def _save(db, title, content, **kw):
    args = ["save", "--type=note", f"--title={title}", f"--content={content}", "--no-embed"]
    for k, v in kw.items():
        args.append(f"--{k.replace('_', '-')}={v}")
    r = run_brain(db, *args)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


# ── invalidate ──────────────────────────────────────────────────────────────

def test_invalidate_hides_from_search_and_recent(db):
    uid = _save(db, "zorbical endpoint moved", "the zorbical endpoint is ep-old")
    r = run_brain(db, "invalidate", uid, "--reason", "endpoint retired")
    assert r.returncode == 0

    r = run_brain(db, "search", "zorbical", "--no-semantic", "--json")
    assert uid not in r.stdout, "invalidated memory leaked into default search"

    r = run_brain(db, "search", "zorbical", "--no-semantic", "--json", "--include-invalid")
    assert uid in r.stdout

    r = run_brain(db, "recent", "100")
    assert uid not in r.stdout


def test_invalidate_undo_and_double_invalidate(db):
    uid = _save(db, "flippable fact", "v1 of the fact")
    assert run_brain(db, "invalidate", uid).returncode == 0
    assert run_brain(db, "invalidate", uid).returncode == 1  # already invalid
    assert run_brain(db, "invalidate", uid, "--undo").returncode == 0
    r = run_brain(db, "search", "flippable", "--no-semantic", "--json")
    assert uid in r.stdout


def test_invalidate_superseded_by_shown_in_get(db):
    old = _save(db, "grobnak config lives in env vars", "old location")
    new = _save(db, "grobnak config lives in SSM", "new location")
    r = run_brain(db, "invalidate", old, "--superseded-by", new)
    assert r.returncode == 0
    r = run_brain(db, "get", old)
    assert "INVALIDATED" in r.stdout and new in r.stdout


def test_as_of_time_travel(db):
    uid = _save(db, "chronofact alpha", "was true once")
    run_brain(db, "invalidate", uid)
    # As of tomorrow: invalidated already → hidden.
    r = run_brain(db, "search", "chronofact", "--no-semantic", "--json", "--as-of", "2099-01-01")
    assert uid not in r.stdout
    # As of long before creation: didn't exist yet → hidden too.
    r = run_brain(db, "search", "chronofact", "--no-semantic", "--json", "--as-of", "2001-01-01")
    assert uid not in r.stdout


# ── link ────────────────────────────────────────────────────────────────────

def test_link_duplicate_does_not_leak_connection(db):
    """Regression: cmd_link left an uncommitted transaction open on the
    IntegrityError path (linking an already-linked pair), which could hold a
    WAL lock and make the NEXT `brain` invocation fail with
    'database is locked'. Two separate `brain link` subprocess calls back to
    back reproduce the original bug (each opens its own connect()).
    """
    a = _save(db, "fact alpha", "alpha body")
    b = _save(db, "fact beta", "beta body")

    r = run_brain(db, "link", a, b, "related_to")
    assert r.returncode == 0, r.stderr

    # Hits the IntegrityError ("link already exists") path in cmd_link.
    r = run_brain(db, "link", a, b, "related_to")
    assert r.returncode == 1
    assert "link already exists" in r.stderr

    # A fresh connect() + write right after must NOT hit "database is locked".
    r = run_brain(db, "link", a, b, "duplicate_of")
    assert r.returncode == 0, r.stderr
    assert "database is locked" not in (r.stderr + r.stdout)


# ── reconcile ───────────────────────────────────────────────────────────────

def test_reconcile_exact_duplicate_noops(db):
    uid = _save(db, "duplicheck fact", "identical body")
    r = run_brain(db, "reconcile", "--title", "duplicheck fact",
                  "--content", "identical body", "--no-semantic")
    assert r.returncode == 0
    packet = json.loads(r.stdout)
    assert packet["suggestion"] == "noop"
    assert packet["target_uid"] == uid


def test_reconcile_auto_add_novel_fact(db):
    r = run_brain(db, "reconcile", "--auto", "--type=note",
                  "--title", "xylograph beacon on port 9151",
                  "--content", "completely novel synthetic fact",
                  "--no-semantic", "--no-embed")
    assert r.returncode == 0, r.stderr
    uid = r.stdout.strip()
    assert len(uid) == 12
    assert uid in run_brain(db, "search", "xylograph", "--no-semantic", "--json").stdout


def test_reconcile_auto_noop_exit_2(db):
    _save(db, "noopable fact", "same body twice")
    r = run_brain(db, "reconcile", "--auto", "--title", "noopable fact",
                  "--content", "same body twice", "--no-semantic")
    assert r.returncode == 2
    assert json.loads(r.stderr)["suggestion"] == "noop"


def test_reconcile_fts_neighbors_suggest_review(db):
    _save(db, "kraken queue dispatcher retries forever", "retry loop bug in dispatcher")
    r = run_brain(db, "reconcile", "--title", "kraken queue dispatcher retry behavior",
                  "--content", "dispatcher retries", "--no-semantic")
    packet = json.loads(r.stdout)
    # Without semantic sims, close FTS neighbors must force a review, never a blind add.
    assert packet["suggestion"] == "review"
    assert packet["neighbors"]


# ── context ─────────────────────────────────────────────────────────────────

def test_context_budget_is_hard_cap(db):
    for i in range(10):
        _save(db, f"plimbork subsystem note {i}",
              "plimbork " + ("filler detail " * 40))
    r = run_brain(db, "context", "plimbork", "--budget", "150", "--no-semantic")
    assert r.returncode == 0
    assert len(r.stdout) <= 150 * 4 + 200  # budget chars + header/footer slack
    assert "brain get" in r.stdout


def test_context_logs_recalls_and_guard_fires(db):
    uid = _save(db, "snerkle rotation policy is every 30 days",
                "the snerkle keys rotate monthly via cron")
    r = run_brain(db, "context", "snerkle rotation", "--no-semantic")
    assert uid in r.stdout
    # recall_log row written → reconcile of the same fact is caught as an
    # exact-dup noop even FTS-only (hash path), and the neighbor is flagged.
    r = run_brain(db, "reconcile", "--title", "snerkle rotation policy is every 30 days",
                  "--content", "the snerkle keys rotate monthly via cron", "--no-semantic")
    packet = json.loads(r.stdout)
    assert packet["suggestion"] == "noop"
    flagged = [n for n in packet["neighbors"] if n["uid"] == uid]
    assert flagged and flagged[0]["recalled_recently"] is True


def test_context_json_shape(db):
    _save(db, "wermble cache eviction", "wermble evicts oldest 100")
    r = run_brain(db, "context", "wermble", "--no-semantic", "--json")
    out = json.loads(r.stdout)
    assert out and {"uid", "title", "abstract", "score"} <= set(out[0])


# ── consolidate ─────────────────────────────────────────────────────────────

def test_consolidate_finds_hash_duplicates_and_merges(db):
    a = _save(db, "quadrupe fact", "same exact body")
    r = run_brain(db, "save", "--type=note", "--title=quadrupe fact",
                  "--content=same exact body", "--no-embed", "--force")
    b = r.stdout.strip()

    r = run_brain(db, "consolidate", "--json")
    clusters = json.loads(r.stdout)
    uids = {m["uid"] for c in clusters for m in c["members"]}
    assert {a, b} <= uids

    r = run_brain(db, "consolidate", "--merge", a, b)
    assert r.returncode == 0
    r = run_brain(db, "get", b)
    assert "INVALIDATED" in r.stdout and a in r.stdout
    # merged dup disappears from default search
    assert b not in run_brain(db, "search", "quadrupe", "--no-semantic", "--json").stdout


def test_consolidate_report_is_read_only(db):
    _save(db, "solo unique fact", "nothing like it")
    before = run_brain(db, "stats").stdout
    run_brain(db, "consolidate")
    assert run_brain(db, "stats").stdout == before


# ── abstract / compact ──────────────────────────────────────────────────────

def test_abstract_saved_and_used_in_context(db):
    uid = _save(db, "vintergon deploy runbook", "very long body " * 50,
                abstract="one-line vintergon summary")
    r = run_brain(db, "context", "vintergon", "--no-semantic")
    assert "one-line vintergon summary" in r.stdout
    # abstract-only update
    r = run_brain(db, "update", uid, "--abstract", "rewritten vintergon abstract")
    assert r.returncode == 0
    assert "rewritten vintergon abstract" in run_brain(db, "context", "vintergon", "--no-semantic").stdout


def test_search_compact_one_line_per_hit(db):
    uid = _save(db, "brachiopod indexing quirk", "details details")
    r = run_brain(db, "search", "brachiopod", "--no-semantic", "--compact")
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    assert len(lines) == 1 and uid in lines[0] and "details" not in lines[0]
