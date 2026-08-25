"""SQLite runtime setup + migrations. One connection per invocation.

Checklist: WAL, foreign_keys=ON, busy_timeout, FTS5-present check,
sqlite-vec load, connection-per-invocation, `user_version` migration runner.

Migration model (v2):
- `schema.sql` is the FROM-EMPTY definition only (fresh install at user_version 2).
- `MIGRATIONS` holds incremental deltas keyed by target version; a store
  upgrading from v1 applies the v2 delta (vec0 1280->1024, FTS triggers, backfill),
  NOT a wholesale re-run of schema.sql (which would crash on existing tables).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Tuple

SCHEMA_VERSION = 2
EMBED_MODEL = "voyage-4-large"  # provider switch 2026-08-25 (ZeroEntropy sunset 09-04)

# Incremental migrations: {target_version: SQL delta}. Applied when a store
# sits at a lower user_version. Each delta must be idempotent to the extent
# that it runs once per version (guarded by PRAGMA user_version in migrate).
MIGRATIONS = {
    2: """
-- v1 -> v2: vec0 1280 -> 1024 (embedding provider: zembed-1 -> voyage-4-large)
--            + FTS5 external-content sync triggers on chunks.
DROP TABLE chunk_embeddings;
CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
  chunk_id integer primary key,
  embedding float[1024]
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', old.id, old.chunk_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, chunk_text) VALUES('delete', old.id, old.chunk_text);
  INSERT INTO chunks_fts(rowid, chunk_text) VALUES (new.id, new.chunk_text);
END;
-- Backfill FTS for chunks written before the triggers existed (v1 stores).
INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild');
""",
}


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
    """Create/upgrade schema. Idempotent: no-op at or past the target version.

    - fresh store (user_version 0): executescript(schema.sql) = full v2 schema
    - existing store below target: apply each incremental MIGRATIONS delta in order
    """
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    if current >= user_version:
        return current
    if current == 0:
        conn.executescript(schema_sql())
    else:
        for version in range(current + 1, user_version + 1):
            delta = MIGRATIONS.get(version)
            if delta is None:
                raise RuntimeError(f"no migration path to schema v{version} from v{current}")
            conn.executescript(delta)
    # SQLite PRAGMA does not bind parameters — integer literal is safe here.
    conn.execute(f"PRAGMA user_version = {int(user_version)}")
    conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('schema_version', ?)", (str(user_version),))
    conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES('embed_model', ?)", (EMBED_MODEL,))
    conn.commit()
    return user_version
