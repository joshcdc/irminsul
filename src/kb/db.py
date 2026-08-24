"""SQLite runtime setup + migrations. One connection per invocation.

Checklist (Phase 0): WAL, foreign_keys=ON, busy_timeout, FTS5-present check,
sqlite-vec load, connection-per-invocation, `user_version` migration runner.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple, Optional

SCHEMA_VERSION = 1


def connect(db_path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def load_vec(conn) -> bool:
    """Load the sqlite-vec extension; True if the vec0 module responds.

    sqlite-vec 0.1.9's `load()` helper omits `enable_load_extension(True)`
    (raises "not authorized" on 3.42.0), so we do the two steps explicitly.
    `vec_version()` takes no arguments.
    """
    try:
        import sqlite_vec  # type: ignore

        conn.enable_load_extension(True)
        conn.load_extension(sqlite_vec.loadable_path())
        conn.execute("SELECT vec_version()")
        return True
    except Exception:
        return False


def fts_ok(conn) -> bool:
    try:
        conn.execute("SELECT rowid FROM chunks_fts LIMIT 1")
        return True
    except sqlite3.DatabaseError:
        return False


def schema_sql() -> str:
    return (Path(__file__).with_name("schema.sql")).read_text(encoding="utf-8")


def migrate(conn, user_version: int = SCHEMA_VERSION) -> int:
    """Create/upgrade schema. Idempotent: no-op at or past the target version."""
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= user_version:
        return current
    conn.executescript(schema_sql())
    # SQLite PRAGMA does not bind parameters — integer literal is safe here.
    conn.execute(f"PRAGMA user_version = {int(user_version)}")
    conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('schema_version', ?)", (str(user_version),))
    conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('embed_model', 'zembed-1')")
    conn.commit()
    return user_version
