"""Bridges FastMCP tool calls to brain.py's argparse-coupled cmd_* functions.

brain.py (../brain-cli/brain.py) is not edited — every accommodation for
"was written as a CLI" lives here:

  1. brain.DB is bound at import time from BRAIN_DB, so the module import
     must be deferred until we know which DB to point at (`_load_brain`).
  2. cmd_* functions write to sys.stdout, which in an MCP stdio server IS the
     JSON-RPC transport. Every call must go through `_run`, which captures
     stdout/stderr so nothing but the tool's own return value crosses the
     wire.
  3. connect() calls sys.exit(...) (not a normal exception) when brain.db is
     missing. `_run` converts that into BrainUnavailable.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import os
from pathlib import Path
from typing import Any, Callable

from brain_mcp.errors import BrainUnavailable

_brain = None  # the imported `brain` module, set by _load_brain()


def _load_brain(db_path: str | None = None):
    """Import brain.py with BRAIN_DB pinned. Must run before any brain use.

    brain.DB = Path(os.environ.get("BRAIN_DB") or ~/brain.db) is evaluated at
    import time, so the env var has to be set before the first `import brain`.
    Safe to call more than once: only the first call actually imports; later
    calls just return the cached module (reassign `brain.DB` directly if you
    need to repoint an already-imported module at a different db).
    """
    global _brain
    if db_path:
        os.environ["BRAIN_DB"] = db_path
    if _brain is None:
        import brain  # noqa: PLC0415 - deliberately deferred, see docstring
        _brain = brain
    return _brain


def get_brain():
    """Return the already-imported brain module, importing with defaults if
    nothing has called _load_brain() yet (BRAIN_DB env var or ~/brain.db)."""
    if _brain is None:
        return _load_brain()
    return _brain


def ensure_db() -> None:
    """Startup preflight: fail loudly now, not on the first tool call."""
    brain = get_brain()
    if not brain.DB.exists():
        raise BrainUnavailable(
            f"brain.db not found at {brain.DB}; run `brain migrate` in brain-cli"
        )
    # Confirm the memories table is actually readable (not just that a file
    # exists at the path — e.g. a zero-byte or unmigrated file).
    try:
        conn = brain.connect()
        conn.execute("SELECT 1 FROM memories LIMIT 1")
        conn.close()
    except SystemExit as e:
        raise BrainUnavailable(str(e.code)) from None


# ────────────────────────────────────────────────────────────────────────────
# Namespace builders — mirror brain.py's build_parser defaults (lines
# 3011-3195) so a future brain-cli flag can never AttributeError via getattr.
# ────────────────────────────────────────────────────────────────────────────


def ns_save(**kw) -> argparse.Namespace:
    brain = get_brain()
    return brain._save_ns(**kw)


def ns_search(
    query: str,
    limit: int = 10,
    type: list[str] | None = None,
    project: str | None = None,
    since_days: int | None = None,
    no_semantic: bool = False,
    explain: bool = False,
    rerank: bool = False,
    rerank_model: str = "haiku",
    compact: bool = False,
    include_invalid: bool = False,
    as_of: str | None = None,
    json: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        query=query, limit=limit, type=type, project=project,
        since_days=since_days, no_semantic=no_semantic, explain=explain,
        rerank=rerank, rerank_model=rerank_model, compact=compact,
        include_invalid=include_invalid, as_of=as_of, json=json,
    )


def ns_context(
    query: str,
    budget: int = 2000,
    limit: int = 20,
    type: list[str] | None = None,
    project: str | None = None,
    since_days: int | None = None,
    no_semantic: bool = False,
    no_blocks: bool = False,
    rerank: bool = False,
    rerank_model: str = "haiku",
    json: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        query=query, budget=budget, limit=limit, type=type, project=project,
        since_days=since_days, no_semantic=no_semantic, no_blocks=no_blocks,
        rerank=rerank, rerank_model=rerank_model, json=json,
    )


def ns_get(uid: str, json: bool = True) -> argparse.Namespace:
    return argparse.Namespace(uid=uid, json=json)


def ns_link(src: str, dst: str, kind: str) -> argparse.Namespace:
    return argparse.Namespace(src=src, dst=dst, kind=kind)


def ns_feedback(uid: str, signal: str, note: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(uid=uid, signal=signal, note=note)


# ────────────────────────────────────────────────────────────────────────────
# The stdout/stderr guard
# ────────────────────────────────────────────────────────────────────────────


def _run(fn: Callable[[Any], int], ns: argparse.Namespace) -> tuple[int, str, str]:
    """Call a brain.py cmd_* function with stdout/stderr captured.

    In an MCP stdio server, real stdout IS the JSON-RPC channel — an
    unguarded cmd_* call would corrupt the protocol stream. This is the only
    place cmd_* functions may be invoked from.
    """
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fn(ns)
    except SystemExit as e:  # connect() when brain.db is missing
        raise BrainUnavailable(str(e.code)) from None
    finally:
        # cmd_* functions open their own sqlite3.Connection via brain.connect()
        # and never explicitly close it. On some error paths (e.g. cmd_link's
        # IntegrityError branch) that connection is left holding an
        # uncommitted transaction with no conn.commit()/rollback()/close() —
        # relying on CPython refcounting to finalize it promptly. Force a
        # collection so the WAL write lock is released before the next
        # brain.connect() call in the same process, instead of racing it.
        gc.collect()
    return rc, out.getvalue(), err.getvalue()
