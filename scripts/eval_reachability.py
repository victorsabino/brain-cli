#!/usr/bin/env python3
"""Reachability eval: can you get a memory back when you DON'T remember its title?

`scripts/eval.py` measures recall on a golden set whose queries look like the
documents. This measures the harder, realer thing: several different phrasings
per memory, including one that deliberately avoids the distinctive proper
nouns — how you actually search months later.

Why it exists: on 2026-07-27 two confident, plausible changes to brain were
measured with this harness and one of them was WRONG.

  - indexing `anchor` to disambiguate clusters   -> recall@5 0.83 to 0.82 (reverted)
  - re-ranking the top-20 window                 -> the vague-query hole

The interesting number was never overall recall (0.83) but the spread across
entry points: keyword 0.98, natural 0.88, vague 0.62. Averages hide the hole.

Probe file — one JSON object per line:

    {"uid": "abc123", "queries": {"natural": "...", "keyword": "...", "vague": "..."}}

The committed probe set (tests/fixtures/probes_synthetic.jsonl) targets the
synthetic fixture, so CI can gate on it. To measure YOUR corpus, generate a
local probe file (see --help-generate); never commit it — it embeds the
titles of your real memories.

    python scripts/eval_reachability.py --db ~/brain.db --probes ~/.config/brain/probes.jsonl
    python scripts/eval_reachability.py --db /tmp/fixture.db \
        --probes tests/fixtures/probes_synthetic.jsonl --min-recall5 0.80
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GENERATE_HELP = """\
To build a probe set for your own corpus (never commit the result):

  1. Sample memories you care about:
       sqlite3 -json ~/brain.db "SELECT uid, title, content, project FROM memories
         WHERE deleted_at IS NULL AND invalid_at IS NULL ORDER BY RANDOM() LIMIT 50"

  2. For each, write three queries that do NOT reuse the title verbatim:
       natural — how you'd ask a colleague, a full question
       keyword — the terse 2-5 word stab you actually type
       vague   — describe the PROBLEM without the distinctive proper nouns
                 (this is the one that finds real gaps; do not just reword)

  3. Emit one JSON object per line:
       {"uid": "...", "queries": {"natural": "...", "keyword": "...", "vague": "..."}}

An LLM does step 2 well in batches of ~25. Keep the file out of git.
"""


def load_brain():
    spec = importlib.util.spec_from_file_location("brain", ROOT / "brain.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="BRAIN_DB to evaluate (default: env/~/brain.db)")
    ap.add_argument("--probes", required=False, help="probe JSONL")
    ap.add_argument("--limit", type=int, default=10, help="retrieval depth (default 10)")
    ap.add_argument("--min-recall5", type=float, help="fail (exit 1) below this overall recall@5")
    ap.add_argument("--min-recall5-vague", type=float,
                    help="fail below this recall@5 on the vague entry point")
    ap.add_argument("--no-semantic", action="store_true", help="FTS only (CI mode)")
    ap.add_argument("--rerank", action="store_true", help="measure WITH the LLM re-ranker")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--help-generate", action="store_true", help="how to build a probe file")
    args = ap.parse_args()

    if args.help_generate:
        print(GENERATE_HELP)
        return 0
    if not args.probes:
        ap.error("--probes is required (see --help-generate)")

    import os
    if args.db:
        os.environ["BRAIN_DB"] = args.db
    brain = load_brain()

    probes = []
    for line in Path(args.probes).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            probes.append(json.loads(line))
    if not probes:
        print("no probes", file=sys.stderr)
        return 2

    conn = brain.connect(load_vec=not args.no_semantic)
    cursor = conn.cursor()
    ns = argparse.Namespace(type=None, project=None, since_days=None,
                            include_invalid=False, as_of=None)
    filter_sql, filter_params = brain._build_filters(conn, ns)

    # The committed synthetic probes key on `title`, because the fixture DB is
    # rebuilt per run and its uids are not stable. Real-corpus probe files use
    # `uid` directly.
    unresolved = []
    for p in probes:
        if p.get("uid"):
            continue
        row = conn.execute(
            "SELECT uid FROM memories WHERE title = ? AND deleted_at IS NULL LIMIT 1",
            (p.get("title", ""),)).fetchone()
        if row:
            p["uid"] = row["uid"]
        else:
            unresolved.append(p.get("title", "?"))
    if unresolved:
        print(f"WARNING: {len(unresolved)} probe(s) matched no memory, skipped:",
              file=sys.stderr)
        for t in unresolved[:5]:
            print(f"  {t[:70]}", file=sys.stderr)
        probes = [p for p in probes if p.get("uid")]
        if not probes:
            return 2

    rows = []
    for p in probes:
        uid = p["uid"]
        for kind, q in (p.get("queries") or {}).items():
            q = (q or "").strip()
            if not q:
                continue
            depth = max(args.limit, brain.RERANK_WINDOW) if args.rerank else args.limit
            top, _ = brain._hybrid_search(conn, cursor, q, depth, filter_sql,
                                          filter_params, no_semantic=args.no_semantic)
            if args.rerank:
                top = brain._rerank(q, top, "haiku", brain.RERANK_WINDOW)
            top = top[:args.limit]
            uids = [r["uid"] for _, r in top]
            rows.append({"uid": uid, "kind": kind, "query": q,
                         "rank": uids.index(uid) + 1 if uid in uids else 0})

    def stats(sel):
        n = len(sel) or 1
        return {
            "n": len(sel),
            "recall@1": sum(1 for x in sel if x["rank"] == 1) / n,
            "recall@5": sum(1 for x in sel if 1 <= x["rank"] <= 5) / n,
            f"recall@{args.limit}": sum(1 for x in sel if x["rank"] >= 1) / n,
            "mrr": sum(1.0 / x["rank"] for x in sel if x["rank"]) / n,
        }

    kinds = sorted({r["kind"] for r in rows})
    overall = stats(rows)
    per_kind = {k: stats([r for r in rows if r["kind"] == k]) for k in kinds}

    if args.json:
        print(json.dumps({"overall": overall, "by_kind": per_kind}, indent=2))
    else:
        def line(label, s):
            print(f"  {label:<12} n={s['n']:<4} " + "  ".join(
                f"{k}={v:.2f}" for k, v in s.items() if k != "n"))
        print(f"reachability — {len(probes)} memories, {len(rows)} queries"
              + (" (reranked)" if args.rerank else ""))
        line("ALL", overall)
        for k in kinds:
            line(k, per_kind[k])
        # An average hides the hole; the WEAKEST entry point is the real score.
        worst = min(per_kind, key=lambda k: per_kind[k]["recall@5"])
        print(f"\n  weakest entry point: {worst} "
              f"(recall@5 {per_kind[worst]['recall@5']:.2f})")

    failed = False
    if args.min_recall5 is not None and overall["recall@5"] < args.min_recall5:
        print(f"FAIL: recall@5 {overall['recall@5']:.3f} < {args.min_recall5}", file=sys.stderr)
        failed = True
    if args.min_recall5_vague is not None and "vague" in per_kind and \
            per_kind["vague"]["recall@5"] < args.min_recall5_vague:
        print(f"FAIL: vague recall@5 {per_kind['vague']['recall@5']:.3f} "
              f"< {args.min_recall5_vague}", file=sys.stderr)
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
