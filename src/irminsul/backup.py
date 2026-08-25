"""Phase 1 — VACUUM INTO consistent snapshots + rotation + recover."""

from __future__ import annotations

import os
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT_GLOB = "irminsul-*.db"


def backup(db_path, backup_dir, keep: int = 10) -> dict:
    """Consistent single-file snapshot via VACUUM INTO (NOT raw `cp`)."""
    backup_dir = Path(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    dest = backup_dir / f"irminsul-{ts}.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        path_lit = str(dest).replace("'", "''")
        conn.execute(f"VACUUM INTO '{path_lit}'")
    finally:
        conn.close()
    removed = rotate(backup_dir, keep)
    return {"snapshot": str(dest), "size": dest.stat().st_size, "kept": keep, "removed": removed}


def rotate(backup_dir, keep: int) -> list:
    files = sorted(Path(backup_dir).glob(SNAPSHOT_GLOB), reverse=True)
    removed = []
    for f in files[keep:]:
        f.unlink(missing_ok=True)
        removed.append(f.name)
    return removed


def resolve_snapshot(backup_dir, target) -> Path:
    """Accept a full path, a bare ts (irminsul-<ts>.db), or a filename."""
    p = Path(target)
    if p.exists():
        return p
    for cand in (Path(backup_dir) / target, Path(backup_dir) / f"irminsul-{target}.db"):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"snapshot not found: {target}")


def recover(db_path, snapshot: Path) -> dict:
    """Replace the whole store from a snapshot (DESTRUCTIVE — callers gate with --yes)."""
    db_path = Path(db_path)
    tmp = Path(str(db_path) + ".recover")
    shutil.copy2(snapshot, tmp)
    os.replace(tmp, db_path)
    for suffix in ("-wal", "-shm"):
        Path(str(db_path) + suffix).unlink(missing_ok=True)
    return {"ok": True, "from": str(snapshot), "to": str(db_path)}
