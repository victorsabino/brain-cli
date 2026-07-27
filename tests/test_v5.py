"""v5 features: feedback-weighted recall, pinned blocks, declared identity,
anchors, and the harvest review-queue drainer.

FTS-only like the rest of the suite — none of these depend on embeddings.
"""
from __future__ import annotations

import json
import sqlite3

from conftest import run_brain


def _save(db, title, content="body text here", **kw):
    args = ["save", "--type", kw.pop("type", "note"), "--title", title, "--content", content]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", v]
    r = run_brain(db, *args)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


# ── schema ──────────────────────────────────────────────────────────────────

def test_migrate_creates_v5_objects(db):
    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    assert {"identity_key", "anchor"} <= cols
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"memory_feedback", "blocks"} <= tables
    conn.close()


# ── feedback-weighted recall ────────────────────────────────────────────────

def test_feedback_records_and_nets_out(db):
    uid = _save(db, "kubernetes ingress timeout root cause")
    r = run_brain(db, "feedback", uid, "up")
    assert r.returncode == 0 and "net +1" in r.stdout
    run_brain(db, "feedback", uid, "up")
    r = run_brain(db, "feedback", uid, "down", "--note", "actually misleading")
    assert "net +1" in r.stdout          # +1 +1 -1
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM memory_feedback").fetchone()[0] == 3
    conn.close()


def test_feedback_unknown_uid_errors(db):
    r = run_brain(db, "feedback", "nosuchuid1234", "up")
    assert r.returncode != 0


def test_feedback_appears_in_explain_only_when_present(db):
    uid = _save(db, "redis eviction policy allkeys-lru", "redis eviction policy notes")
    r = run_brain(db, "search", "redis eviction", "--explain", "--json", "--no-semantic")
    hit = json.loads(r.stdout)[0]
    assert hit["explain"]["feedback_net"] == 0
    assert hit["explain"]["feedback_bonus"] == 0.0

    run_brain(db, "feedback", uid, "up")
    r = run_brain(db, "search", "redis eviction", "--explain", "--json", "--no-semantic")
    hit = json.loads(r.stdout)[0]
    assert hit["explain"]["feedback_net"] == 1
    assert hit["explain"]["feedback_bonus"] > 0


def test_feedback_bonus_is_capped(db):
    """A brigaded memory must not be able to outrank real relevance."""
    uid = _save(db, "postgres vacuum tuning", "postgres vacuum tuning content")
    for _ in range(50):
        run_brain(db, "feedback", uid, "up")
    r = run_brain(db, "search", "postgres vacuum", "--explain", "--json", "--no-semantic")
    assert json.loads(r.stdout)[0]["explain"]["feedback_bonus"] <= 0.08


def test_downvote_demotes_but_does_not_hide(db):
    uid = _save(db, "flaky auth test quarantine", "flaky auth test quarantine content")
    for _ in range(5):
        run_brain(db, "feedback", uid, "down")
    r = run_brain(db, "search", "flaky auth quarantine", "--json", "--no-semantic")
    hits = json.loads(r.stdout)
    assert any(h["uid"] == uid for h in hits), "downvoted memory must still be findable"


# ── pinned core blocks ──────────────────────────────────────────────────────

def test_block_set_list_rm(db):
    r = run_brain(db, "block", "set", "persona", "Terse. No preamble.")
    assert r.returncode == 0
    r = run_brain(db, "block", "list")
    assert "persona" in r.stdout and "Terse" in r.stdout
    r = run_brain(db, "block", "rm", "persona")
    assert "removed" in r.stdout
    assert "no pinned blocks" in run_brain(db, "block", "list").stdout


def test_block_truncates_to_char_limit(db):
    run_brain(db, "block", "set", "big", "x" * 500, "--char-limit", "50")
    conn = sqlite3.connect(db)
    val = conn.execute("SELECT value FROM blocks WHERE label='big'").fetchone()[0]
    conn.close()
    assert len(val) == 50


def test_blocks_injected_into_context(db):
    _save(db, "deploy runbook for corena", "run bin/deploy then verify")
    run_brain(db, "block", "set", "rules", "Never push to main.")
    r = run_brain(db, "context", "deploy runbook", "--no-semantic")
    assert "## Pinned (brain)" in r.stdout and "Never push to main." in r.stdout
    r = run_brain(db, "context", "deploy runbook", "--no-semantic", "--no-blocks")
    assert "Pinned" not in r.stdout


def test_pinned_blocks_cannot_eat_more_than_half_the_budget(db):
    _save(db, "some matched memory", "matched memory content")
    for i in range(12):
        run_brain(db, "block", "set", f"b{i}", "y" * 300, "--char-limit", "300")
    r = run_brain(db, "context", "matched memory", "--budget", "200", "--no-semantic")
    pinned_chars = sum(len(ln) for ln in r.stdout.splitlines() if ln.startswith("- [b"))
    assert pinned_chars <= (200 * 4) // 2


def test_context_json_stays_a_list_with_pinned_flag(db):
    """Shape is load-bearing: agents already parse this as a list."""
    _save(db, "json shape memory", "json shape content")
    run_brain(db, "block", "set", "persona", "be terse")
    out = json.loads(run_brain(db, "context", "json shape", "--no-semantic", "--json").stdout)
    assert isinstance(out, list)
    assert out[0]["pinned"] is True and out[0]["type"] == "block"
    assert any(e.get("uid") for e in out)


# ── declared identity ───────────────────────────────────────────────────────

def test_identity_key_is_deterministic_and_namespaced(brain_mod):
    a = brain_mod.identity_key("note", "Corena Deploy Runbook")
    b = brain_mod.identity_key("note", "  corena   deploy runbook ")
    assert a == b, "normalization must make these the same key"
    assert brain_mod.identity_key("decision", "corena deploy runbook") != a
    assert brain_mod.identity_key("note", "") == ""


def test_identity_forces_update_over_similarity(db):
    first = _save(db, "corena deploy runbook", "step one: bin/deploy",
                  identity="corena deploy runbook")
    # Deliberately different wording — similarity alone would say "add".
    r = run_brain(db, "reconcile", "--type", "note",
                  "--title", "how we ship corena to production",
                  "--content", "totally different phrasing about shipping",
                  "--identity", "corena deploy runbook", "--no-semantic")
    packet = json.loads(r.stdout)
    assert packet["suggestion"] == "update"
    assert packet["target_uid"] == first
    assert "by declaration" in packet["reason"]


def test_identity_absent_leaves_behaviour_unchanged(db):
    """No --identity → the v4 path is untouched. Text must share NO tokens with
    the stored memory: in FTS-only mode any lexical neighbor has sim=None,
    which correctly routes to 'review' (existing v4 behaviour, not a v5 change)."""
    _save(db, "unrelated topic alpha", "kubernetes ingress annotations")
    r = run_brain(db, "reconcile", "--type", "note", "--title", "zqx wibble frobnicator",
                  "--content", "marmalade dirigible tuesday", "--no-semantic")
    assert json.loads(r.stdout)["suggestion"] == "add"


def test_identity_unique_among_live_memories(db):
    _save(db, "first holder", "a", identity="the one true fact")
    conn = sqlite3.connect(db)
    key = conn.execute(
        "SELECT identity_key FROM memories WHERE title='first holder'").fetchone()[0]
    assert key and key.startswith("note:")
    conn.close()


# ── anchors ─────────────────────────────────────────────────────────────────

def test_anchor_saved_and_grouped_in_context(db):
    _save(db, "corena ships on fridays", "corena release cadence", anchor="corena")
    _save(db, "corena uses vercel edge config", "corena acl storage", anchor="corena")
    _save(db, "falkor runs on ecs", "falkor infra", anchor="falkor")
    r = run_brain(db, "context", "corena falkor", "--no-semantic")
    assert "### corena" in r.stdout
    assert "### falkor" in r.stdout


def test_context_flat_when_nothing_anchored(db):
    _save(db, "plain memory one", "plain one")
    _save(db, "plain memory two", "plain two")
    r = run_brain(db, "context", "plain memory", "--no-semantic")
    assert "###" not in r.stdout


# ── review queue ────────────────────────────────────────────────────────────

def _queue(tmp_path, entries):
    p = tmp_path / "review.jsonl"
    p.write_text("".join(json.dumps(e) + "\n" for e in entries))
    return p


def _entry(title, sim, content="queued candidate content"):
    return {
        "candidate": {"type": "note", "title": title, "content": content,
                      "project": "p", "tags": ["t"], "abstract": "a"},
        "packet": {"suggestion": "review", "neighbors": [{"uid": "x", "sim": sim,
                                                          "title": "neighbor"}]},
    }


def test_review_empty_queue(db, tmp_path):
    p = tmp_path / "none.jsonl"
    r = run_brain(db, "review", "--file", str(p))
    assert r.returncode == 0 and "empty" in r.stdout


def test_review_summarises_buckets(db, tmp_path):
    p = _queue(tmp_path, [_entry("near dup", 0.95), _entry("ambiguous", 0.80),
                          _entry("looks new", 0.10)])
    r = run_brain(db, "review", "--file", str(p))
    assert "3 candidate(s)" in r.stdout
    assert "near-dup" in r.stdout and "ambiguous" in r.stdout


def test_review_skips_malformed_lines(db, tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps(_entry("good one", 0.5)) + "\n{not json\n")
    r = run_brain(db, "review", "--file", str(p))
    assert r.returncode == 0 and "1 candidate(s)" in r.stdout


def test_review_resolve_drop_removes_entry(db, tmp_path):
    p = _queue(tmp_path, [_entry("first", 0.5), _entry("second", 0.5)])
    r = run_brain(db, "review", "--file", str(p), "--resolve", "0", "--action", "drop")
    assert r.returncode == 0 and "1 left" in r.stdout
    remaining = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert len(remaining) == 1
    assert remaining[0]["candidate"]["title"] == "second"


def test_review_resolve_save_persists_memory(db, tmp_path):
    p = _queue(tmp_path, [_entry("promote me to a real memory", 0.5)])
    r = run_brain(db, "review", "--file", str(p), "--resolve", "0", "--action", "save")
    assert r.returncode == 0
    got = run_brain(db, "search", "promote me real memory", "--json", "--no-semantic")
    assert any("promote me" in h["title"] for h in json.loads(got.stdout))
    assert p.read_text().strip() == ""


def test_review_resolve_out_of_range(db, tmp_path):
    p = _queue(tmp_path, [_entry("only", 0.5)])
    assert run_brain(db, "review", "--file", str(p), "--resolve", "7",
                     "--action", "drop").returncode != 0


def test_review_clear_requires_yes(db, tmp_path):
    p = _queue(tmp_path, [_entry("a", 0.5), _entry("b", 0.5)])
    assert run_brain(db, "review", "--file", str(p), "--clear").returncode != 0
    assert len(p.read_text().splitlines()) == 2
    r = run_brain(db, "review", "--file", str(p), "--clear", "--yes")
    assert r.returncode == 0 and p.read_text().strip() == ""


def test_review_auto_dry_run_changes_nothing(db, tmp_path):
    p = _queue(tmp_path, [_entry("brand new unrelated fact xyzzy", 0.1)])
    before = p.read_text()
    r = run_brain(db, "review", "--file", str(p), "--auto", "--dry-run", "--no-semantic")
    assert r.returncode == 0 and "DRY RUN" in r.stdout
    assert p.read_text() == before


def test_review_auto_drops_exact_duplicates(db, tmp_path):
    """A queued candidate identical to a live memory is a dup, not a decision."""
    _save(db, "already known fact", "already known body")
    p = _queue(tmp_path, [_entry("already known fact", 0.99, content="already known body")])
    r = run_brain(db, "review", "--file", str(p), "--auto", "--no-semantic")
    assert r.returncode == 0
    assert "dropped 1" in r.stdout
    assert p.read_text().strip() == ""


def test_review_auto_drops_titleless_entries(db, tmp_path):
    p = _queue(tmp_path, [{"candidate": {"title": "  ", "content": "x"}, "packet": {}}])
    r = run_brain(db, "review", "--file", str(p), "--auto", "--no-semantic")
    assert "dropped 1" in r.stdout and p.read_text().strip() == ""


# ── durability judge (review --auto --judge) ────────────────────────────────

def _stub_claude(tmp_path, payload: str, exit_code: int = 0):
    """Fake `claude` that prints `payload` regardless of args."""
    import os
    import stat
    bindir = tmp_path / "stub-bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / "claude"
    exe.write_text(f"#!/bin/sh\ncat > /dev/null 2>&1 || true\n"
                   f"printf '%s' {json.dumps(payload)}\nexit {exit_code}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return {"PATH": f"{bindir}:{os.environ['PATH']}"}


def _fresh(n):
    """n candidates with no lexical overlap with each other or the db."""
    words = ["zorble", "quimbly", "fnordic", "blivet", "gronk", "wibbly"]
    return [_entry(f"{words[i % len(words)]}{i} unique subject line", 0.05,
                   content=f"{words[i % len(words)]}{i} durable body detail")
            for i in range(n)]


def test_judge_drops_items_marked_junk(db, tmp_path):
    p = _queue(tmp_path, _fresh(3))
    verdicts = json.dumps([{"i": 0, "verdict": "junk"}, {"i": 1, "verdict": "usable"},
                           {"i": 2, "verdict": "junk"}])
    r = run_brain(db, "review", "--file", str(p), "--auto", "--judge", "--no-semantic",
                  env_extra=_stub_claude(tmp_path, verdicts))
    assert r.returncode == 0, r.stderr
    assert "saved 1" in r.stdout
    assert "durability judge" in r.stdout


def test_judge_dry_run_writes_nothing(db, tmp_path):
    p = _queue(tmp_path, _fresh(2))
    before = p.read_text()
    verdicts = json.dumps([{"i": 0, "verdict": "junk"}, {"i": 1, "verdict": "usable"}])
    r = run_brain(db, "review", "--file", str(p), "--auto", "--judge", "--dry-run",
                  "--no-semantic", env_extra=_stub_claude(tmp_path, verdicts))
    assert "DRY RUN" in r.stdout and p.read_text() == before


def test_judge_fails_safe_on_bad_json(db, tmp_path):
    """A model hiccup must never silently delete facts."""
    p = _queue(tmp_path, _fresh(2))
    r = run_brain(db, "review", "--file", str(p), "--auto", "--judge", "--dry-run",
                  "--no-semantic", env_extra=_stub_claude(tmp_path, "not json at all"))
    assert r.returncode == 0
    assert "would drop 0" in r.stdout and "save 2" in r.stdout


def test_judge_fails_safe_on_nonzero_exit(db, tmp_path):
    p = _queue(tmp_path, _fresh(2))
    r = run_brain(db, "review", "--file", str(p), "--auto", "--judge", "--dry-run",
                  "--no-semantic", env_extra=_stub_claude(tmp_path, "[]", exit_code=1))
    assert "would drop 0" in r.stdout and "save 2" in r.stdout


def test_judge_ignores_out_of_range_indices(db, tmp_path):
    """A model echoing bogus indices must not drop unrelated candidates."""
    p = _queue(tmp_path, _fresh(2))
    verdicts = json.dumps([{"i": 99, "verdict": "junk"}, {"i": -1, "verdict": "junk"}])
    r = run_brain(db, "review", "--file", str(p), "--auto", "--judge", "--dry-run",
                  "--no-semantic", env_extra=_stub_claude(tmp_path, verdicts))
    assert "would drop 0" in r.stdout and "save 2" in r.stdout


def test_auto_without_judge_does_not_call_llm(db, tmp_path):
    """No --judge → no model dependency at all (stub exits 1; must not matter)."""
    p = _queue(tmp_path, _fresh(2))
    r = run_brain(db, "review", "--file", str(p), "--auto", "--dry-run", "--no-semantic",
                  env_extra=_stub_claude(tmp_path, "boom", exit_code=1))
    assert r.returncode == 0 and "save 2" in r.stdout
    assert "durability judge" not in r.stdout


# ── v6: anchor is actually INDEXED (v5 stored it but indexed it nowhere) ────

def test_fts_indexes_anchor(db):
    conn = sqlite3.connect(db)
    ddl = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name='memories_fts'").fetchone()[0]
    conn.close()
    assert "anchor" in ddl, "anchor must be an FTS column or it can never be retrieved"


def test_anchor_is_searchable(db):
    """The whole point of v6: find a memory by its anchor alone."""
    _save(db, "shortened the page title", "changed the h1 text", anchor="corena")
    r = run_brain(db, "search", "corena", "--json", "--no-semantic")
    hits = json.loads(r.stdout)
    assert any("shortened the page title" in h["title"] for h in hits), \
        "anchor token must retrieve the memory even though it is not in title/content"


def test_anchor_reaches_embedded_chunk_text(brain_mod):
    texts = brain_mod.chunk_texts("some title", "some body", anchor="corena")
    assert all("corena" in t for t in texts)
    plain = brain_mod.chunk_texts("some title", "some body")
    assert all("corena" not in t for t in plain)


def test_anchor_update_refreshes_fts(db):
    """Anchor set after insert must reach FTS via the trigger, not just the row."""
    _save(db, "zzz obscure title", "zzz obscure body")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE memories SET anchor='falkor' WHERE title='zzz obscure title'")
    conn.commit()
    conn.close()
    r = run_brain(db, "search", "falkor", "--json", "--no-semantic")
    assert any("zzz obscure" in h["title"] for h in json.loads(r.stdout))


def test_anchor_backfill_sets_and_skips(db, tmp_path):
    _save(db, "alpha memory", "alpha body")
    _save(db, "beta memory", "beta body")
    verdicts = json.dumps([{"i": 0, "anchor": "Corena"}, {"i": 1, "anchor": None}])
    r = run_brain(db, "anchor", "--no-embed",
                  env_extra=_stub_claude(tmp_path, verdicts))
    assert r.returncode == 0, r.stderr
    assert "set 1 anchor" in r.stdout and "1 had no single entity" in r.stdout
    conn = sqlite3.connect(db)
    vals = [x[0] for x in conn.execute(
        "SELECT anchor FROM memories WHERE anchor IS NOT NULL")]
    conn.close()
    assert vals == ["corena"], "anchors must be normalized lowercase for cluster consistency"


def test_anchor_backfill_rejects_vague_buckets(db, tmp_path):
    """A wrong anchor is worse than none — it merges unrelated clusters."""
    _save(db, "gamma memory", "gamma body")
    for bad in ("general", "misc", "N/A"):
        r = run_brain(db, "anchor", "--no-embed",
                      env_extra=_stub_claude(tmp_path, json.dumps([{"i": 0, "anchor": bad}])))
        assert "set 0 anchor" in r.stdout, f"{bad!r} should be rejected"


def test_anchor_backfill_dry_run_writes_nothing(db, tmp_path):
    _save(db, "delta memory", "delta body")
    r = run_brain(db, "anchor", "--dry-run", "--no-embed",
                  env_extra=_stub_claude(tmp_path, json.dumps([{"i": 0, "anchor": "corena"}])))
    assert "would set 1" in r.stdout
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM memories WHERE anchor IS NOT NULL").fetchone()[0] == 0
    conn.close()


def test_anchor_backfill_fails_safe(db, tmp_path):
    _save(db, "epsilon memory", "epsilon body")
    r = run_brain(db, "anchor", "--no-embed", env_extra=_stub_claude(tmp_path, "garbage"))
    assert r.returncode == 0 and "1 left unanchored" in r.stdout
