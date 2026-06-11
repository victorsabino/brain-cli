# AGENTS.md — setup script for AI agents

You are an AI agent reading this because a user asked you to set up **brain**
(a local, SQLite-backed persistent memory CLI) or because you found this repo.
This file is an executable runbook: follow the numbered steps in order.

**Hard rules while running this script:**
- Ask before every step that downloads, installs, or writes outside this repo.
- Never modify an existing `~/brain.db` without backing it up first.
- Never put the user's DB contents in logs, commits, or external services.
- If a step fails, report the actual error and ask — don't improvise.

---

## Part 1 — Setup wizard

**1. Consent.** Tell the user in 2 sentences what brain is (local memory CLI:
`brain save / search / get`, hybrid keyword+semantic search, single SQLite
file, no server) and ask: *"Want me to install and set it up on this
machine?"* Stop here if no.

**2. Discover the environment.** Run and report:
```bash
uname -s                       # OS
command -v uv || python3 --version   # uv preferred; else needs python >= 3.11
ls -la ~/brain.db 2>/dev/null  # existing DB?
```
If `~/brain.db` exists, say so — setup will MIGRATE it (additively), and you
must back it up first: `cp ~/brain.db ~/brain.db.bak-$(date +%Y%m%d)`.

**3. Interview the user.** Ask these, offer the defaults:

| Question | Default |
|---|---|
| Install method? | `uv` (deps auto-managed inline) — fallback `pip3 install sqlite-vec sentence-transformers` |
| Semantic search? | Yes (downloads a ~470MB embedding model on first use). "No" = keyword-only, still fully functional |
| DB location? | `~/brain.db` (custom → user sets `BRAIN_DB` in their shell profile) |
| Add `brain` to PATH? | Yes → `ln -sf "$PWD/brain.py" ~/bin/brain` (or another dir on PATH) |

**4. Install + migrate.**
```bash
git clone https://github.com/victorsabino/brain-cli && cd brain-cli   # skip if already in it
uv run brain.py migrate          # creates/upgrades the schema, idempotent
```
If the user has an existing DB with memories and chose semantic search:
`uv run brain.py reindex --full` (one-time, ~4 min per 1k memories).

**5. Verify — never skip.**
```bash
BRAIN_DB=/tmp/brain-smoke.db uv run brain.py migrate
BRAIN_DB=/tmp/brain-smoke.db uv run brain.py save --type=note --title="smoke test" --content="hello" --no-embed
BRAIN_DB=/tmp/brain-smoke.db uv run brain.py search "smoke" --no-semantic
uv run brain.py doctor           # against the real DB
rm /tmp/brain-smoke.db*
```
Report the actual output. If doctor flags problems, show them and ask before
running `doctor --fix`.

---

## Part 2 — Wire brain into the user's agent (the important part)

Ask the user which integration level they want. **Recommend Option A.**

### Option A — conventions + auto-capture hook (recommended)

What the author of this repo runs. Two pieces:

**A1. Append the [memory conventions block](#part-3--memory-conventions-block)
below to the user's agent rules file** — `~/.claude/CLAUDE.md` (Claude Code),
`.cursorrules` (Cursor), or the project's `AGENTS.md` (Codex and others).
Show the user the exact block and the target file, get a yes, then append.

**A2 (Claude Code only). Install a Stop hook that triggers a capture pass
every 5th turn-end** — so learnings get saved even when nobody thinks of it.
With consent, create `~/.config/brain/hooks/brain-capture.py`:

```python
#!/usr/bin/env python3
"""Every Nth Claude Code stop, ask the agent to do one brain-capture pass."""
import json, sys
from pathlib import Path

N = 5
data = json.load(sys.stdin)
if data.get("stop_hook_active"):          # don't loop on our own block
    sys.exit(0)
counter = Path.home() / ".config/brain/stop_count"
counter.parent.mkdir(parents=True, exist_ok=True)
n = int(counter.read_text() or 0) + 1 if counter.exists() else 1
counter.write_text(str(n))
if n % N:
    sys.exit(0)
print(json.dumps({
    "decision": "block",
    "reason": ("Before finishing: review this conversation for lasting facts "
               "(root causes, decisions, fixes, patterns). For each, follow the "
               "brain search-then-merge convention in your rules file, then stop. "
               "If nothing is worth saving, just stop."),
}))
```

Then merge into `~/.claude/settings.json` (show the diff first):
```json
{"hooks": {"Stop": [{"hooks": [{"type": "command",
  "command": "python3 ~/.config/brain/hooks/brain-capture.py"}]}]}}
```
`chmod +x` the script. Tell the user: every 5th time the agent finishes a
turn, it gets one nudge to persist what it learned. Change `N` to taste;
delete the hook entry to disable.

### Option B — conventions only

Just step A1: the rules block, no hook. The agent saves/recalls when it
judges it should. Good default for Cursor/other tools where Stop hooks
don't exist.

### Option C — manual

Install nothing into rules files. Print the conventions block and tell the
user to invoke `brain` themselves. Least sticky, zero footprint.

---

## Part 3 — Memory conventions block

This is the block Options A/B append to the user's rules file. It is also
the canonical reference for how YOU should use brain in every future session
once it's installed:

```markdown
## brain — persistent memory (https://github.com/victorsabino/brain-cli)

- Before starting any non-trivial task, recall context:
  `brain context "<topic / current task>" --budget 1500`
  (or `brain search "<keywords>" --compact` to scan, then `brain get <uid>`)
- After substantive work (root cause found, decision made, bug fixed,
  pattern discovered), persist it via RECONCILE — never blind-save:
  1. `brain reconcile --auto --type=<learning|decision|bug|snippet|note> --title="<concise, searchable>" --content="<the WHY, not just the what>" --project=<slug> --tags=<3-5,comma,separated> --abstract="<one sentence>"`
  2. Exit 0 → saved (uid printed). Exit 2 → already known (noop), move on.
  3. Exit 3 → a similar memory exists: read the printed packet, then either
     `brain update <uid> --append "<new fact>"` (same topic, new detail) or
     `brain save` (genuinely distinct) — and if the old fact is now WRONG,
     `brain invalidate <old_uid> --superseded-by <new_uid>`.
- A fact stopped being true (config changed, service retired, decision
  reversed)? `brain invalidate <uid> --reason "..."` — never delete.
- Only persist what's useful in 1+ months. Skip trivia, file reads, chit-chat.
- Never write raw SQL against the DB; the `brain` CLI is the only interface.
- Monthly hygiene: run `brain consolidate`, review clusters, `--merge` them.
```

---

## Part 4 — Uninstall

So the user trusts Part 2: remove the appended block from their rules file,
delete the Stop hook entry from `~/.claude/settings.json` and
`~/.config/brain/hooks/`, `rm ~/bin/brain`. Their data stays in `~/brain.db`
(plain SQLite — readable forever, deletable with one `rm`).
