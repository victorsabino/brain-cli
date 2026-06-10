#!/usr/bin/env python3
"""Build a synthetic brain.db fixture + golden set for the search eval.

All data here is INVENTED (fake cities, fake projects, fake bugs) — safe to
commit, unlike the real golden set at ~/.config/brain/golden.jsonl.

Flow: create the pre-v3 base schema → run scripts/migrate.py (BRAIN_DB) →
insert ~40 memories through `brain.py save --no-embed` (exercises the real
save path + FTS triggers, no ML deps) → write golden_synthetic.jsonl with the
uids generated for THIS fixture build.

Usage:
  python3 tests/fixtures/make_fixture.py /tmp/brain-fixture.db
  python3 scripts/eval.py --golden tests/fixtures/golden_synthetic.jsonl \
      --db /tmp/brain-fixture.db --no-semantic --min-recall5 1.0
"""

from __future__ import annotations
import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
BRAIN = ROOT / "brain.py"
MIGRATE = ROOT / "scripts" / "migrate.py"
GOLDEN_OUT = HERE / "golden_synthetic.jsonl"

# Mirrors the real pre-v3 base schema (migrate.py ALTERs the rest on top).
BASE_SQL = """
CREATE TABLE memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  uid TEXT UNIQUE NOT NULL,
  type TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT,
  project TEXT,
  area TEXT,
  tags TEXT,
  source_file TEXT,
  status TEXT,
  priority TEXT,
  energy TEXT,
  points INTEGER,
  metadata TEXT,
  created_at DATETIME DEFAULT (datetime('now')),
  updated_at DATETIME DEFAULT (datetime('now')),
  completed_at DATETIME
);
CREATE TABLE stats (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at DATETIME DEFAULT (datetime('now'))
);
"""

LONG_PIPELINE_LOG = """Tuning log for the Harborlight ingestion pipeline (parquet lake behind federated search).

Baseline: 40 min full-refresh for Port Vesper + Quartzdale corpora combined. Bottleneck was small-file explosion — 18k parquet files under 2MB after every nightly crawl.

First pass: batched the writer to 128MB row groups, cut file count 30x. Refresh dropped to 22 min but query planners still scanned cold footers.

Second pass: partitioned by city slug instead of crawl date. Pruning now skips 90% of footers for single-city queries.

- [2026-04-02] moved manifest generation off the critical path; refresh 22 -> 16 min
- [2026-04-18] bumped writer parallelism to 8, saturates the NVMe scratch volume
- [2026-05-06] switched parquet compaction to zstd level 7 — 31% smaller lake, decode cost negligible on read path"""

# (type, title, content, project, tags)
MEMORIES: list[tuple[str, str, str, str, str]] = [
    ("bug", "Port Vesper kiosk reboot loop after firmware 4.2 rollout",
     "Symptom: lobby kiosks power-cycle every ~90s after the 4.2 firmware push.\n\nCause: watchdog timeout shrank from 120s to 60s while the boot splash still takes 75s on the old SoC.\n\n- [2026-03-11] mitigation: pinned fleet to firmware 4.1\n- [2026-03-14] vendor patch extends watchdog to 180s, verified on 3 units",
     "harborlight", "kiosk,firmware"),
    ("bug", "Quartzdale permit portal 502 on PDF uploads over 8MB",
     "Nginx ingress buffers the whole body in memory; client_max_body_size is 8m while the app allows 25MB. Large permit PDFs bounce with 502 before reaching the app.",
     "quartzdale-ada", "permits,nginx"),
    ("bug", "Sablecrest transit API returns stale bus positions after midnight UTC",
     "The position cache key embeds the service date computed in local time, but the feed flips at midnight UTC — between 00:00Z and local midnight every lookup hits yesterday's key. Buses appear frozen.",
     "tidepool-app", "transit,cache"),
    ("bug", "Marrowgate alert siren cron fires twice on DST transition",
     "The siren test cron is scheduled in local time; on fall-back the 01:30 slot occurs twice and the siren self-test runs twice. Citizens reported double sirens.\n\nFix: schedule in UTC, render local time only for display.",
     "marrowgate-ops", "cron,dst"),
    ("bug", "kelp-crm webhook signature mismatch when payload has emoji",
     "HMAC was computed over the UTF-16 string length on the vendor side and UTF-8 bytes on ours. Any payload with emoji (multi-byte) fails signature verification. Vendor fixed; we also normalize to raw bytes before HMAC.",
     "kelp-crm", "webhook,hmac"),
    ("learning", "Harborlight chat widget drops websocket on cellular handoff — heartbeat fix",
     "Wifi→LTE handoff silently kills the socket without a close frame; the widget waited forever. A 25s ping/pong heartbeat plus exponential reconnect makes the drop invisible to users.",
     "harborlight", "websocket,mobile"),
    ("learning", "Tidepool push notifications throttled by FCM when topic fanout exceeds 500",
     "FCM topic sends above ~500 devices/sec get silently queued for minutes. Batched device-group sends with explicit rate limiting restored sub-second delivery.",
     "tidepool-app", "push,fcm"),
    ("learning", "Quartzdale ADA scanner false positives on decorative SVG icons",
     "The scanner flagged every inline SVG without a title element. Decorative icons with aria-hidden=true are exempt under WCAG; the rule now checks aria-hidden and role=presentation before flagging.",
     "quartzdale-ada", "wcag,svg"),
    ("learning", "Sablecrest geocoder confuses Pier 9 with Pier 90 — anchor the number match",
     "Prefix matching on the landmark index returned Pier 90 for 'Pier 9'. Anchoring the numeric token with a word boundary fixed it; added 40 landmark regression cases.",
     "tidepool-app", "geocoding,regex"),
    ("learning", "Postgres BRIN beats btree for Marrowgate sensor telemetry inserts",
     "Telemetry is append-only and time-correlated; a BRIN index on observed_at is 400x smaller than btree and insert overhead is near zero. Range scans for dashboards stayed fast.",
     "marrowgate-ops", "postgres,brin"),
    ("decision", "Adopt RRF over weighted-sum for Harborlight federated search",
     "Weighted-sum required constant re-normalization between bm25 magnitudes and cosine sims; every source change broke the scale. Rank-based fusion is scale-free.",
     "harborlight", "search,rrf"),
    ("decision", "kelp-crm: route email ingest through SQS instead of direct lambda invoke",
     "Direct invoke lost mail during deploys. SQS gives retry with DLQ and absorbs vendor bursts; latency budget allows the extra hop.",
     "kelp-crm", "sqs,email"),
    ("decision", "Tidepool app ships offline-first sqlite cache instead of live queries",
     "Transit riders lose connectivity underground; live queries meant blank screens. The app now reads a local sqlite replica synced opportunistically; freshness indicator shows sync age.",
     "tidepool-app", "offline,sqlite"),
    ("decision", "Quartzdale picks Fastify over Express for the permit API rewrite",
     "Schema validation is first-class and the JSON serializer is 2-3x faster on the permit list endpoint. Express middleware ecosystem wasn't a factor — we use 4 middlewares total.",
     "quartzdale-ada", "fastify,api"),
    ("snippet", "imagemagick recipe: trim scanned permit margins and deskew",
     "magick scan.png -deskew 40% -fuzz 8% -trim +repage clean.png\n\nDeskew before trim — trim on a skewed scan eats real content at the corners.",
     "quartzdale-ada", "imagemagick"),
    ("snippet", "jq one-liner to diff two Harborlight config exports",
     "jq -S . a.json > /tmp/a; jq -S . b.json > /tmp/b; diff -u /tmp/a /tmp/b\n\nSort keys first or every diff is noise.",
     "harborlight", "jq,config"),
    ("snippet", "sqlite window function for sessionizing kiosk taps",
     "SELECT *, SUM(new_session) OVER (PARTITION BY kiosk_id ORDER BY ts) AS session_id FROM (SELECT *, CASE WHEN ts - LAG(ts) OVER (PARTITION BY kiosk_id ORDER BY ts) > 300 THEN 1 ELSE 0 END AS new_session FROM taps);",
     "harborlight", "sqlite,sessions"),
    ("note", "Port Vesper stakeholder map — who signs off on kiosk deployments",
     "Facilities owns physical placement, IT owns the image, the city clerk signs the content policy. All three must approve before a kiosk goes to a new building.",
     "harborlight", "stakeholders"),
    ("note", "Marrowgate siren maintenance window calendar",
     "Self-tests first Wednesday monthly at 11:00 local. No deploys to the siren controller within 24h of a test window.",
     "marrowgate-ops", "maintenance"),
    ("note", "Tidepool beta tester cohort demographics",
     "412 testers: 60% daily transit riders, 25% accessibility-tool users, 15% city staff. Recruit more night-shift riders — after-midnight flows are under-tested.",
     "tidepool-app", "beta"),
    ("task", "Rotate kelp-crm webhook secrets before the Q3 vendor audit",
     "Both directions: our signing key and the vendor's. Coordinate a 1h overlap window where both keys validate.",
     "kelp-crm", "security"),
    ("task", "Backfill Sablecrest transit positions for the March feed outage",
     "Vendor can replay 2026-03-04 → 2026-03-09 from their archive. Dedup on (vehicle_id, ts) before insert.",
     "tidepool-app", "backfill"),
    ("task", "Write runbook for Quartzdale portal 502 mitigation",
     "Steps: confirm ingress buffer limit, raise client_max_body_size, page on sustained 502 rate > 1%.",
     "quartzdale-ada", "runbook"),
    ("person", "Mira Voss — Port Vesper ops lead, prefers async voice notes",
     "Owns kiosk fleet ops. Hates long Slack threads; send a 60s voice note or a 3-bullet summary. Decision-maker for firmware rollouts.",
     "harborlight", "ops"),
    ("person", "Dario Klee — kelp-crm vendor contact, slow on email",
     "Escalate via the shared vendor channel, not email (3-4 day email latency). Knows the webhook signing internals personally.",
     "kelp-crm", "vendor"),
    ("project", "Harborlight — federated city search platform overview",
     "One search box over city sites, permits, transit alerts and kiosk content for Port Vesper and Quartzdale. Components: crawler, parquet lake, federated ranker, chat widget.",
     "harborlight", "overview"),
    ("project", "Tidepool — resident mobile app charter",
     "Offline-first companion app for Sablecrest residents: transit, service requests, alerts. KPI: task completion without connectivity.",
     "tidepool-app", "charter"),
    ("learning", "Harborlight ingestion pipeline tuning log", LONG_PIPELINE_LOG,
     "harborlight", "parquet,performance"),
    ("bug", "Harborlight dedup drops legit pages when titles differ only by accent",
     "Title normalization strips diacritics before hashing, so 'Café Permit FAQ' and 'Cafe Permit FAQ' collapse into one page. Hash the raw title; use the normalized form for display only.",
     "harborlight", "dedup,unicode"),
    ("decision", "Marrowgate keeps on-prem postgres, rejects cloud migration for siren control",
     "Siren control must survive a regional internet outage. Telemetry analytics may move to cloud later; the control plane stays on-prem with a satellite backup link.",
     "marrowgate-ops", "onprem"),
    ("snippet", "curl + hmac example to replay a kelp-crm webhook locally",
     "sig=$(printf '%s' \"$body\" | openssl dgst -sha256 -hmac \"$secret\" -hex | cut -d' ' -f2); curl -X POST localhost:3000/hooks/kelp -H \"X-Kelp-Signature: $sig\" -d \"$body\"",
     "kelp-crm", "curl,hmac"),
    ("note", "Quartzdale council meeting notes 2026-03 — permit fee changes",
     "Council approved tiered permit fees effective July. Portal needs a fee calculator before the effective date; legal wants the old fee table archived, not deleted.",
     "quartzdale-ada", "council"),
    ("learning", "Sablecrest fare engine rounding: banker's rounding broke reconciliation",
     "The fare engine used round-half-even while the settlement vendor rounds half-up. Daily reconciliation drifted by a few cents per thousand taps. Aligned on half-up and added a reconciliation invariant test.",
     "tidepool-app", "rounding,fares"),
    ("task", "Spike: evaluate meilisearch for Harborlight typo tolerance",
     "Bm25 misses one-letter typos in city service names. Timebox 2 days; measure recall delta on the crawler corpus.",
     "harborlight", "spike,search"),
    ("note", "Tidepool app store review checklist",
     "Screenshots per device class, privacy nutrition labels, offline-mode demo video, accessibility statement link.",
     "tidepool-app", "release"),
    ("bug", "Tidepool deep links open a blank screen when the app cold-starts on Android 15",
     "The link intent arrives before the JS bridge is ready and the route is dropped. Queue the initial intent and replay it after the navigator mounts. Warm-start was never affected.",
     "tidepool-app", "deeplinks,android"),
    ("learning", "Quartzdale scanner: aria-hidden focusable elements are the top real issue",
     "After killing the SVG false positives, 70% of remaining valid findings are focusable elements inside aria-hidden containers — keyboard users tab into invisible UI.",
     "quartzdale-ada", "wcag,focus"),
    ("decision", "Port Vesper kiosks get a read-only Friday deploy freeze",
     "Weekend support coverage is one person; kiosk regressions found Saturday wait until Monday. Content updates stay allowed, binaries freeze Friday 12:00.",
     "harborlight", "deploys"),
    ("snippet", "python script to replay Marrowgate siren cron events safely",
     "for ev in load_events(day): dispatch(ev, dry_run=True)\n\nAlways dry-run first; the dispatcher has no idempotency key.",
     "marrowgate-ops", "python,cron"),
    ("note", "kelp-crm field mapping cheatsheet — invoice vs estimate objects",
     "Invoices: total_cents, due_at, account_ref. Estimates: same shape but amounts are advisory; never sync estimates into the ledger table.",
     "kelp-crm", "mapping"),
]

# (query, index into MEMORIES, note) — distinctive tokens so FTS-only hits.
GOLDEN_QUERIES: list[tuple[str, int, str]] = [
    ("kiosk stuck rebooting after firmware update", 0, "paraphrase"),
    ("permit uploads failing with 502 on large pdf files", 1, "paraphrase"),
    ("bus positions stale after midnight", 2, "paraphrase"),
    ("siren cron fires twice during dst", 3, "paraphrase"),
    ("webhook signature mismatch emoji payload", 4, "paraphrase"),
    ("websocket drops during cellular handoff heartbeat", 5, "paraphrase"),
    ("decorative svg icons false positives accessibility scanner", 7, "paraphrase"),
    ("brin versus btree for sensor telemetry", 9, "paraphrase"),
    ("offline-first sqlite cache for the mobile app", 12, "decision"),
    ("parquet compaction zstd level", 27, "tail: dated bullet at END of long log"),
    ("deep links blank screen on cold start android", 35, "paraphrase"),
    ("banker's rounding broke fare reconciliation", 32, "paraphrase"),
]


def main() -> int:
    p = argparse.ArgumentParser(description="Build synthetic brain.db fixture + golden set")
    p.add_argument("db", help="output DB path (overwritten if present)")
    p.add_argument("--golden-out", default=str(GOLDEN_OUT))
    args = p.parse_args()

    db = Path(args.db).expanduser()
    if db.exists():
        db.unlink()
    db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db)
    conn.executescript(BASE_SQL)
    conn.commit()
    conn.close()

    env = {**os.environ, "BRAIN_DB": str(db)}
    r = subprocess.run([sys.executable, str(MIGRATE)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"✗ migrate failed: {r.stderr}{r.stdout}")

    uids: list[str] = []
    for mtype, title, content, project, tags in MEMORIES:
        r = subprocess.run(
            [sys.executable, str(BRAIN), "save", f"--type={mtype}", f"--title={title}",
             f"--content={content}", f"--project={project}", f"--tags={tags}", "--no-embed"],
            capture_output=True, text=True, env=env,
        )
        if r.returncode != 0:
            sys.exit(f"✗ save failed for {title!r}: {r.stderr}")
        uids.append(r.stdout.strip().splitlines()[-1])

    golden = Path(args.golden_out)
    with golden.open("w") as f:
        for query, idx, note in GOLDEN_QUERIES:
            f.write(json.dumps(
                {"query": query, "expected_uid": uids[idx], "note": note},
                ensure_ascii=False) + "\n")

    print(f"✓ fixture: {db} ({len(MEMORIES)} memories)")
    print(f"✓ golden:  {golden} ({len(GOLDEN_QUERIES)} queries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
