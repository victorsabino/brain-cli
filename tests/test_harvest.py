"""brain harvest: transcript → candidates → reconcile. Hermetic — the
`claude` CLI is stubbed with a tiny script on PATH, no network, no model."""
from __future__ import annotations

import json
import os
import stat

from conftest import run_brain


def _stub_claude(tmp_path, payload: str):
    """Fake `claude` executable that prints `payload` regardless of args."""
    bindir = tmp_path / "stub-bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / "claude"
    exe.write_text(f"#!/bin/sh\ncat > /dev/null 2>&1 || true\nprintf '%s' {json.dumps(payload)}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    return {"PATH": f"{bindir}:{os.environ['PATH']}"}


def _transcript(tmp_path, turns):
    """Write a minimal Claude Code-style JSONL transcript."""
    p = tmp_path / "session.jsonl"
    lines = []
    for role, text in turns:
        content = text if role == "user" else [{"type": "text", "text": text}]
        lines.append(json.dumps({"type": role, "message": {"role": role, "content": content}}))
    p.write_text("\n".join(lines) + "\n")
    return p


CANDIDATE = json.dumps([{
    "type": "learning",
    "title": "grimblewok cache stampede fixed with jittered TTL",
    "content": "the grimblewok cache stampeded on expiry; fix was +-10% TTL jitter",
    "project": "testproj", "tags": "cache,ttl", "abstract": "jittered TTL fixes stampede",
}])


def test_harvest_extracts_and_saves(db, tmp_path):
    t = _transcript(tmp_path, [("user", "why does grimblewok stampede? " + "x" * 2000),
                               ("assistant", "found it: TTL expiry sync. fixed with jitter.")])
    r = run_brain(db, "harvest", str(t), "--no-semantic", env_extra=_stub_claude(tmp_path, CANDIDATE))
    assert r.returncode == 0, r.stderr
    assert "1 added" in r.stdout
    assert "grimblewok" in run_brain(db, "search", "grimblewok stampede", "--no-semantic").stdout


def test_harvest_watermark_no_reprocess(db, tmp_path):
    t = _transcript(tmp_path, [("user", "grimblewok question " + "x" * 2500)])
    env = _stub_claude(tmp_path, CANDIDATE)
    r1 = run_brain(db, "harvest", str(t), "--no-semantic", env_extra=env)
    assert "1 added" in r1.stdout
    # Second run: zero new bytes → skip, nothing re-extracted, no duplicate.
    r2 = run_brain(db, "harvest", str(t), "--no-semantic", env_extra=env)
    assert "skipped" in r2.stdout
    out = json.loads(run_brain(db, "search", "grimblewok", "--no-semantic", "--json").stdout)
    assert len(out) == 1  # no duplicate row from the second run


def test_harvest_duplicate_candidate_noops(db, tmp_path):
    env = _stub_claude(tmp_path, CANDIDATE)
    t1 = _transcript(tmp_path, [("user", "first session " + "x" * 2500)])
    run_brain(db, "harvest", str(t1), "--no-semantic", env_extra=env)
    # New transcript, same extracted fact → reconcile catches the exact dup.
    t2 = tmp_path / "session2.jsonl"
    t2.write_text(t1.read_text())
    r = run_brain(db, "harvest", str(t2), "--no-semantic", env_extra=env)
    assert "0 added" in r.stdout and "1 noop" in r.stdout


def test_harvest_small_delta_accumulates(db, tmp_path):
    t = _transcript(tmp_path, [("user", "tiny message")])
    env = _stub_claude(tmp_path, "[]")
    r = run_brain(db, "harvest", str(t), "--no-semantic", env_extra=env)
    assert "skipped" in r.stdout
    # watermark must NOT have advanced — appending more later still sees old bytes
    with open(t, "a") as f:
        content = [{"type": "text", "text": "big follow-up " + "y" * 2500}]
        f.write(json.dumps({"type": "assistant", "message": {"content": content}}) + "\n")
    r = run_brain(db, "harvest", str(t), "--no-semantic", env_extra=env)
    assert "0 candidate(s)" in r.stdout  # stub returns [], but extraction RAN on full delta


def test_harvest_failed_extraction_keeps_watermark(db, tmp_path):
    t = _transcript(tmp_path, [("user", "important fact " + "x" * 2500)])
    bindir = tmp_path / "stub-bin"
    bindir.mkdir()
    exe = bindir / "claude"
    exe.write_text("#!/bin/sh\nexit 1\n")
    exe.chmod(0o755)
    env = {"PATH": f"{bindir}:{os.environ['PATH']}"}
    r = run_brain(db, "harvest", str(t), "--no-semantic", env_extra=env)
    assert r.returncode == 1
    # retry with a working stub must still see the bytes
    r = run_brain(db, "harvest", str(t), "--no-semantic",
                  env_extra=_stub_claude(tmp_path, CANDIDATE))
    assert "1 added" in r.stdout


def test_harvest_recursion_guard(db, tmp_path):
    t = _transcript(tmp_path, [("user", "anything " + "x" * 2500)])
    r = run_brain(db, "harvest", str(t), "--no-semantic",
                  env_extra={**_stub_claude(tmp_path, CANDIDATE), "BRAIN_HARVEST": "1"})
    assert r.returncode == 0 and "added" not in r.stdout


def test_harvest_tool_noise_excluded(db, tmp_path):
    p = tmp_path / "noisy.jsonl"
    rows = [
        {"type": "user", "message": {"content": "real user question " + "q" * 2400}},
        {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "secret_dump"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "content": "SECRETVALUE123"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "the answer"}]}},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    r = run_brain(db, "harvest", str(p), "--dry-run", "--no-semantic")
    assert "real user question" in r.stdout and "the answer" in r.stdout
    assert "SECRETVALUE123" not in r.stdout and "secret_dump" not in r.stdout
