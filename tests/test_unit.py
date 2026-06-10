"""Unit tests on brain.py loaded as a module: chunking helpers + connect()."""
from __future__ import annotations

import re

import pytest


def _no_ws(s: str) -> str:
    return re.sub(r"\s+", "", s)


def test_split_chunks_loses_no_content(brain_mod):
    """Splitting may renormalize whitespace but must never drop characters."""
    text = "\n\n".join(
        [f"paragraph {i} with some distinctive content token{i}" for i in range(30)]
        + ["x" * 2500]  # pathological para — exercises the hard split
    )
    chunks = brain_mod._split_chunks(text)
    assert _no_ws("".join(chunks)) == _no_ws(text)


def test_split_chunks_respects_hard_cap(brain_mod):
    text = "y" * (brain_mod.CHUNK_HARD * 5 + 123)
    chunks = brain_mod._split_chunks(text)
    assert all(len(c) <= brain_mod.CHUNK_HARD for c in chunks)
    assert _no_ws("".join(chunks)) == text


def test_split_chunks_empty_content(brain_mod):
    assert brain_mod._split_chunks("") == [""]


def test_chunk_texts_prepends_title_to_every_chunk(brain_mod):
    title = "tuning log"
    content = "\n\n".join(f"para {i} " + "z" * 600 for i in range(5))
    texts = brain_mod.chunk_texts(title, content)
    assert len(texts) > 1
    assert all(t.startswith(title + "\n") for t in texts)


def test_chunk_texts_title_only_when_no_content(brain_mod):
    assert brain_mod.chunk_texts("just a title", "") == ["just a title"]


def test_chunk_texts_enforces_chunk_cap(brain_mod):
    """rowid = memory_id * CHUNK_CAP + idx — more than CHUNK_CAP chunks would
    collide with the next memory's rowid range."""
    content = "\n\n".join("w" * 600 for _ in range(brain_mod.CHUNK_CAP * 2))
    texts = brain_mod.chunk_texts("big", content)
    assert len(texts) == brain_mod.CHUNK_CAP


def test_connect_sets_wal_and_busy_timeout(brain_mod, db, monkeypatch):
    monkeypatch.setattr(brain_mod, "DB", db)
    conn = brain_mod.connect()
    try:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        conn.close()


def test_connect_missing_db_exits(brain_mod, tmp_path, monkeypatch):
    monkeypatch.setattr(brain_mod, "DB", tmp_path / "nope.db")
    with pytest.raises(SystemExit):
        brain_mod.connect()
