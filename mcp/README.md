# brain-mcp

Experimental MCP server wrapping [brain-cli](https://github.com/victorsabino/brain-cli)'s
`brain.py` (personal knowledge DB at `~/brain.db`, or `$BRAIN_DB`) as 6 MCP
tools: `brain_save`, `brain_search`, `brain_get`, `brain_context`,
`brain_link`, `brain_feedback`. Built with [FastMCP](https://gofastmcp.com).

`brain.py` itself is not modified. All adaptation (stdout-capture guard,
Namespace synthesis, error shaping) lives in `brain_mcp/adapter.py` and
`brain_mcp/server.py`.

Status: local prototype, both transports smoke-tested manually. Streamable-http
requires a bearer token (see Auth below); stdio has no auth. No deployment yet.

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

# streamable-http (requires BRAIN_MCP_TOKEN — see Auth below)
BRAIN_MCP_TRANSPORT=http BRAIN_MCP_HOST=127.0.0.1 BRAIN_MCP_PORT=8000 \
  BRAIN_MCP_TOKEN=<your-token> uv run --directory . brain-mcp
# serves at http://127.0.0.1:8000/mcp
```

`BRAIN_DB` overrides the default `~/brain.db` — set it before starting the
server.

## Auth

This server is meant to be called only by clients you control (Claude Code, or
a server you run) — never registered as a Claude.ai hosted "connector" — so it
uses a single static bearer token rather than full OAuth.

- **stdio** (default transport): no auth. It's a local subprocess with no
  network exposure; a token would be meaningless.
- **streamable-http**: requires `BRAIN_MCP_TOKEN` to be set. If
  `BRAIN_MCP_TRANSPORT=http` is set without `BRAIN_MCP_TOKEN`, the server
  refuses to start rather than silently serving unauthenticated HTTP.

Generate your own token and keep it out of version control (e.g. in a
`mcp/.env` you `chmod 600` and source before running — `.env`/`*.env` are
gitignored repo-wide):

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Clients authenticate by sending `Authorization: Bearer <token>`. With
`fastmcp.Client`, pass the token string as `auth`:

```python
from fastmcp import Client

async with Client("http://127.0.0.1:8000/mcp", auth="<your-token>") as client:
    result = await client.call_tool("brain_search", {"query": "..."})
```

## Out of scope (CLI-only, not exposed as MCP tools)

migrate, reindex, consolidate, delete, update, invalidate, harvest, review,
reconcile, anchor, artifact, secrets, block, doctor, stats, recent, history,
tags.
