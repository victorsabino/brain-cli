#!/usr/bin/env python3
"""brain search-quality eval. Stdlib-only so CI can run it without ML deps.

Reads a golden set (JSONL: {"query": str, "expected_uid": str, "note"?: str}),
runs `brain search --json --limit 10` per query in a subprocess, and scores
recall@1, recall@5, MRR@10. Exits 1 when recall@5 drops below --min-recall5 —
this is the regression gate that was missing when 4 scoring bugs shipped
silently.

Usage:
  python3 scripts/eval.py --golden ~/.config/brain/golden.jsonl
  python3 scripts/eval.py --golden tests/fixtures/golden_synthetic.jsonl \
      --db /tmp/fixture.db --no-semantic --min-recall5 1.0
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BRAIN = Path(__file__).resolve().parent.parent / "brain.py"
LIMIT = 10  # ranks beyond 10 count as MISS — matches MRR@10


def load_golden(path: Path) -> list[dict]:
    cases = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        obj = json.loads(line)
        if not obj.get("query") or not obj.get("expected_uid"):
            sys.exit(f"✗ {path}:{i} missing query/expected_uid")
        cases.append(obj)
    if not cases:
        sys.exit(f"✗ {path}: empty golden set")
    return cases


def brain_cmd(no_semantic: bool) -> list[str]:
    """Prefer uv (resolves sqlite-vec + model deps); plain python works for
    FTS-only runs because brain.py guards every heavy import."""
    if not no_semantic and shutil.which("uv"):
        return ["uv", "run", str(BRAIN)]
    return [sys.executable, str(BRAIN)]


def run_query(cmd: list[str], query: str, env: dict, no_semantic: bool) -> list[str]:
    args = cmd + ["search", query, "--json", f"--limit={LIMIT}"]
    if no_semantic:
        args.append("--no-semantic")
    r = subprocess.run(args, capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        sys.exit(f"✗ search failed for {query!r}: {r.stderr.strip()}")
    try:
        return [hit["uid"] for hit in json.loads(r.stdout)]
    except (json.JSONDecodeError, KeyError, TypeError):
        sys.exit(f"✗ unparseable search output for {query!r}: {r.stdout[:200]}")


def main() -> int:
    p = argparse.ArgumentParser(description="Eval brain search quality against a golden set")
    p.add_argument("--golden", required=True, help="path to golden JSONL")
    p.add_argument("--db", help="DB path (sets BRAIN_DB for the subprocess)")
    p.add_argument("--no-semantic", action="store_true", help="FTS-only (CI mode, no ML deps)")
    p.add_argument("--min-recall5", type=float, default=0.7, help="gate: exit 1 below this")
    args = p.parse_args()

    cases = load_golden(Path(args.golden).expanduser())
    env = dict(os.environ)  # BRAIN_DB passthrough when --db absent
    if args.db:
        env["BRAIN_DB"] = str(Path(args.db).expanduser())
    cmd = brain_cmd(args.no_semantic)

    hits1 = hits5 = 0
    rr_sum = 0.0
    print(f"{'rank':>4}  {'expected':<14} query")
    print("-" * 78)
    for c in cases:
        uids = run_query(cmd, c["query"], env, args.no_semantic)
        rank = uids.index(c["expected_uid"]) + 1 if c["expected_uid"] in uids else None
        if rank:
            hits1 += rank == 1
            hits5 += rank <= 5
            rr_sum += 1.0 / rank
        mark = f"{rank:>4}" if rank else "MISS"
        note = f"  [{c['note']}]" if c.get("note") else ""
        print(f"{mark}  {c['expected_uid']:<14} {c['query'][:55]}{note}")

    n = len(cases)
    r1, r5, mrr = hits1 / n, hits5 / n, rr_sum / n
    print("-" * 78)
    print(f"n={n}  recall@1={r1:.3f}  recall@5={r5:.3f}  MRR@10={mrr:.3f}")
    if r5 < args.min_recall5:
        print(f"✗ recall@5 {r5:.3f} < gate {args.min_recall5}", file=sys.stderr)
        return 1
    print(f"✓ recall@5 {r5:.3f} >= gate {args.min_recall5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
