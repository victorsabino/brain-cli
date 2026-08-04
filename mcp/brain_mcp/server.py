"""FastMCP stdio server exposing 6 tools over brain.py's personal knowledge DB."""
from __future__ import annotations

import json as _json
import os

from fastmcp import FastMCP

from brain_mcp import adapter
from brain_mcp.errors import BrainError, BrainInvalidInput

mcp = FastMCP("brain-mcp")


def _not_found(uid: str) -> dict:
    return {
        "found": False,
        "uid": uid,
        "message": f"no memory with uid {uid}; uids come from brain_search or brain_context results",
    }


@mcp.tool
def brain_save(
    title: str,
    content: str,
    type: str = "note",
    project: str | None = None,
    area: str | None = None,
    tags: list[str] | None = None,
    abstract: str | None = None,
    anchor: str | None = None,
    identity: str | None = None,
    source_file: str | None = None,
    force: bool = False,
    status: str | None = None,
    priority: str | None = None,
    energy: str | None = None,
    points: int | None = None,
    due_at: str | None = None,
    external_ref: str | None = None,
    parent_uid: str | None = None,
) -> dict:
    """Store a durable fact in the personal knowledge base so it survives this session. `type` is one of learning, decision, bug, snippet, note, task, person, project. Write `content` as the WHY behind a fact, not a transcript. `abstract` is a one-sentence summary — it is the only line `brain_context` injects, so write it. `anchor` is the entity the fact is about (a client, repo, person, system) and groups related lessons. Identical title+content is rejected as a duplicate and returns the existing uid; use `force=true` to override. Returns the new uid."""
    brain = adapter.get_brain()
    if type not in brain.CANONICAL_TYPES:
        raise BrainInvalidInput(
            f"unknown type '{type}'. Valid: {sorted(brain.CANONICAL_TYPES)}"
        )
    ns = adapter.ns_save(
        title=title, content=content, type=type,
        project=project or "", area=area or "", tags=tags or [],
        abstract=abstract, anchor=anchor, identity=identity,
        source_file=source_file or "", force=force,
        status=status, priority=priority, energy=energy, points=points,
        due_at=due_at, external_ref=external_ref, parent_uid=parent_uid,
    )
    rc, out, err = adapter._run(brain.cmd_save, ns)
    if rc == 0:
        return {"status": "saved", "uid": out.strip()}
    if rc == 2:
        return {
            "status": "duplicate",
            "uid": out.strip(),
            "message": "identical title+content already stored; pass force=true to save anyway",
        }
    raise BrainInvalidInput(err.strip().lstrip("✗ ").strip() or err.strip())


@mcp.tool
def brain_search(
    query: str,
    limit: int = 10,
    type: list[str] | None = None,
    project: str | None = None,
    since_days: int | None = None,
    compact: bool = False,
    include_invalid: bool = False,
    as_of: str | None = None,
    explain: bool = False,
) -> dict:
    """Search the personal knowledge base by meaning and keyword together (BM25 + embeddings, fused). Use half-remembered phrasing freely — it matches on meaning, not exact words. Returns uid + title + a 300-char snippet, ranked; call `brain_get` for the full body of the few that matter. Filter with `type`, `project`, or `since_days`. Invalidated (superseded) memories are hidden unless `include_invalid=true`; `as_of="YYYY-MM-DD"` time-travels to what was believed true then. An empty query lists the newest memories matching the filters."""
    brain = adapter.get_brain()
    ns = adapter.ns_search(
        query=query, limit=limit, type=type, project=project,
        since_days=since_days, compact=compact,
        include_invalid=include_invalid, as_of=as_of, explain=explain,
        no_semantic=False, rerank=False, json=True,
    )
    rc, out, err = adapter._run(brain.cmd_search, ns)
    if rc != 0:
        raise BrainError(err.strip() or f"brain search failed (rc={rc})")
    results = _json.loads(out) if out.strip() else []
    return {
        "query": query,
        "count": len(results),
        "results": results,
        "semantic": bool(brain._vec_loaded),
        "truncated": len(results) == limit,
    }


@mcp.tool
def brain_get(uid: str) -> dict:
    """Fetch one memory in full by its uid (from `brain_search` or `brain_context`). Returns the complete content plus tags, project, anchor, and whether it has been invalidated or superseded by a newer memory. Reading a memory also records that it was useful, which feeds search ranking — so prefer this over re-searching for something you already located."""
    brain = adapter.get_brain()
    ns = adapter.ns_get(uid=uid, json=True)
    rc, out, err = adapter._run(brain.cmd_get, ns)
    if rc != 0:
        return _not_found(uid)
    return _json.loads(out)


@mcp.tool
def brain_context(
    query: str,
    budget: int = 2000,
    limit: int = 20,
    type: list[str] | None = None,
    project: str | None = None,
    since_days: int | None = None,
    include_blocks: bool = True,
) -> dict:
    """Get a compact, token-budgeted briefing of everything the knowledge base knows that is relevant to a topic — one-line abstracts, not full bodies — plus any pinned always-on facts. Call this once at the start of work on a project or task, using the project name or task description as the query, then `brain_get` the specific uids that matter. `budget` is a hard cap in tokens (default 2000) so this can never flood the context window."""
    if not query.strip():
        raise BrainInvalidInput("query required (e.g. the project name or current task)")
    brain = adapter.get_brain()
    ns = adapter.ns_context(
        query=query, budget=budget, limit=limit, type=type, project=project,
        since_days=since_days, no_semantic=False,
        no_blocks=not include_blocks, rerank=False, json=True,
    )
    rc, out, err = adapter._run(brain.cmd_context, ns)
    if rc != 0:
        raise BrainError(err.strip() or f"brain context failed (rc={rc})")
    out = out.strip()
    if not out.startswith("["):
        # cmd_context prints plain text ("(no brain memories matched: ...)")
        # even in --json mode when nothing matched.
        return {"query": query, "budget_tokens": budget, "count": 0, "memories": [], "blocks": []}
    raw = _json.loads(out)
    blocks = [e for e in raw if e.get("pinned")]
    memories = [e for e in raw if not e.get("pinned")]
    return {
        "query": query,
        "budget_tokens": budget,
        "blocks": blocks,
        "memories": memories,
        "count": len(memories),
    }


@mcp.tool
def brain_link(src: str, dst: str, kind: str) -> dict:
    """Record a typed relationship between two stored memories by uid. `kind` is one of: cites, caused_by, fixed_by, superseded_by, blocks, related_to, duplicate_of. Use it after saving a fix to point at the bug it fixed, or after replacing an outdated fact. Re-linking an existing pair is a no-op, not an error."""
    brain = adapter.get_brain()
    if kind not in brain.LINK_KINDS:
        raise BrainInvalidInput(f"invalid kind '{kind}'. Valid: {sorted(brain.LINK_KINDS)}")
    ns = adapter.ns_link(src=src, dst=dst, kind=kind)
    rc, out, err = adapter._run(brain.cmd_link, ns)
    if rc == 0:
        return {"status": "linked", "src": src, "dst": dst, "kind": kind}
    err = err.strip()
    if "already exists" in err:
        return {"status": "exists", "src": src, "dst": dst, "kind": kind}
    if "not found" in err:
        conn = brain.connect()
        missing = []
        if not conn.execute("SELECT 1 FROM memories WHERE uid = ?", (src,)).fetchone():
            missing.append(src)
        if not conn.execute("SELECT 1 FROM memories WHERE uid = ?", (dst,)).fetchone():
            missing.append(dst)
        conn.close()
        return {
            "found": False,
            "missing_uids": missing,
            "message": f"uid(s) not found: {', '.join(missing)}",
        }
    raise BrainError(err or f"brain link failed (rc={rc})")


@mcp.tool
def brain_feedback(uid: str, signal: str, note: str | None = None) -> dict:
    """Tell the knowledge base that a memory you just used was genuinely useful (`up`) or misleading (`down`). This adjusts its search ranking slightly. Use `down` for low-quality/off-target; a factually WRONG memory should instead be superseded with a corrected `brain_save` + `brain_link(kind="superseded_by")`. `note` records why."""
    if signal not in ("up", "down"):
        raise BrainInvalidInput(f"invalid signal '{signal}'. Valid: ['up', 'down']")
    brain = adapter.get_brain()
    ns = adapter.ns_feedback(uid=uid, signal=signal, note=note)
    rc, out, err = adapter._run(brain.cmd_feedback, ns)
    if rc != 0:
        return _not_found(uid)
    conn = brain.connect()
    row = conn.execute(
        "SELECT COALESCE(SUM(f.signal), 0) FROM memory_feedback f "
        "JOIN memories m ON m.id = f.memory_id WHERE m.uid = ?",
        (uid,),
    ).fetchone()
    conn.close()
    net = row[0] if row else 0
    return {"status": "recorded", "uid": uid, "signal": signal, "net": net}


def main() -> None:
    adapter._load_brain(os.environ.get("BRAIN_DB"))
    adapter.ensure_db()
    transport = os.environ.get("BRAIN_MCP_TRANSPORT", "stdio")
    if transport == "http":
        host = os.environ.get("BRAIN_MCP_HOST", "127.0.0.1")
        port = int(os.environ.get("BRAIN_MCP_PORT", "8000"))
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
