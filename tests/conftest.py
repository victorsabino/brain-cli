"""Shared fixtures for the hermetic test suite.

Every DB is built from scratch under pytest tmp dirs (pre-v3 base schema →
scripts/migrate.py via the BRAIN_DB seam). ~/brain.db is NEVER opened — the
old smoke tests copied the live DB, which leaked real data into assertions
and broke on anyone else's machine.

The suite runs FTS-only by default: tests that need the embedding model are
gated on `requires_model` (sentence-transformers absent in CI by design).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BRAIN = ROOT / "brain.py"
MIGRATE = ROOT / "scripts" / "migrate.py"
EVAL = ROOT / "scripts" / "eval.py"
MAKE_FIXTURE = ROOT / "tests" / "fixtures" / "make_fixture.py"

HAS_ST = importlib.util.find_spec("sentence_transformers") is not None
HAS_VEC = importlib.util.find_spec("sqlite_vec") is not None

requires_model = pytest.mark.skipif(
    not (HAS_ST and HAS_VEC), reason="sentence-transformers/sqlite-vec not installed (CI mode)"
)
requires_vec = pytest.mark.skipif(not HAS_VEC, reason="sqlite-vec not installed")


def load_module(name: str, path: Path):
    """Import a script file as a module (brain.py / make_fixture.py have no package)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_brain(db: Path, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run brain.py as a subprocess against `db` (the real CLI surface)."""
    env = {**os.environ, "BRAIN_DB": str(db), **(env_extra or {})}
    return subprocess.run(
        [sys.executable, str(BRAIN), *args],
        capture_output=True, text=True, timeout=300, env=env,
    )


def run_migrate(db: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "BRAIN_DB": str(db)}
    return subprocess.run(
        [sys.executable, str(MIGRATE), *args],
        capture_output=True, text=True, timeout=120, env=env,
    )


def create_base_db(path: Path) -> None:
    """Write the pre-v3 base schema (memories + stats) — migrate.py's input."""
    fixture_mod = load_module("make_fixture_schema", MAKE_FIXTURE)
    conn = sqlite3.connect(path)
    conn.executescript(fixture_mod.BASE_SQL)
    conn.commit()
    conn.close()


@pytest.fixture(scope="session")
def brain_mod():
    """brain.py loaded as a module — unit tests for pure helpers (_split_chunks…)."""
    return load_module("brain", BRAIN)


@pytest.fixture(scope="session")
def base_db(tmp_path_factory) -> Path:
    """One empty migrated-v3 DB per session; mutation tests copy it (fast)."""
    path = tmp_path_factory.mktemp("base") / "brain.db"
    create_base_db(path)
    r = run_migrate(path)
    assert r.returncode == 0, f"migrate failed: {r.stderr}{r.stdout}"
    return path


@pytest.fixture()
def db(base_db, tmp_path) -> Path:
    """Per-test throwaway copy of the migrated empty DB."""
    dest = tmp_path / "brain.db"
    shutil.copy2(base_db, dest)
    return dest


@pytest.fixture(scope="session")
def fixture_db(tmp_path_factory) -> tuple[Path, Path]:
    """(db, golden) built by make_fixture.py — synthetic corpus for search tests.

    Session-scoped and treated as read-only: search never mutates rows
    (access_count bumps only on `get`), so sharing it is safe.
    """
    d = tmp_path_factory.mktemp("fixture")
    db_path = d / "fixture.db"
    golden = d / "golden_synthetic.jsonl"
    r = subprocess.run(
        [sys.executable, str(MAKE_FIXTURE), str(db_path), f"--golden-out={golden}"],
        capture_output=True, text=True, timeout=600,
    )
    assert r.returncode == 0, f"make_fixture failed: {r.stderr}{r.stdout}"
    return db_path, golden
