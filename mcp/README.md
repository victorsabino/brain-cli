# brain-mcp

Experimental MCP server wrapping [brain-cli](https://github.com/victorsabino/brain-cli)'s
`brain.py` (personal knowledge DB at `~/brain.db`, or `$BRAIN_DB`) as 6 MCP
tools: `brain_save`, `brain_search`, `brain_get`, `brain_context`,
`brain_link`, `brain_feedback`. Built with [FastMCP](https://gofastmcp.com).

`brain.py` itself is not modified. All adaptation (stdout-capture guard,
Namespace synthesis, error shaping) lives in `brain_mcp/adapter.py` and
`brain_mcp/server.py`.

Status: local prototype, both transports smoke-tested manually. No auth, no
deployment yet.

This is nested inside [brain-cli](https://github.com/victorsabino/brain-cli)
as the `mcp/` subdirectory — `brain.py` one level up is not modified; all
adaptation lives here in `brain_mcp/`.

## Setup

`mcp/pyproject.toml` path-depends on the parent directory (`..`), where
`brain.py` lives:

```bash
git clone https://github.com/victorsabino/brain-cli.git
cd brain-cli/mcp && uv sync
```

## Run

```bash
# stdio (default)
uv run --directory . brain-mcp
# or, for the inspector:
fastmcp dev brain_mcp/server.py

# streamable-http
BRAIN_MCP_TRANSPORT=http BRAIN_MCP_HOST=127.0.0.1 BRAIN_MCP_PORT=8000 \
  uv run --directory . brain-mcp
# serves at http://127.0.0.1:8000/mcp
```

`BRAIN_DB` overrides the default `~/brain.db` — set it before starting the
server.

## Out of scope (CLI-only, not exposed as MCP tools)

migrate, reindex, consolidate, delete, update, invalidate, harvest, review,
reconcile, anchor, artifact, secrets, block, doctor, stats, recent, history,
tags.
