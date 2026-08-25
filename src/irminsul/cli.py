"""irminsul — Typer CLI command surface.

Phase 1: init, doctor, help, schema, config, put, get, list, stats,
delete, restore, prune, backup, recover, import, export.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import typer

from . import backup
from . import config as cfg
from . import db
from . import io as kbio
from . import pages

app = typer.Typer(no_args_is_help=True, help="irminsul — agent knowledge base (irminsul-io v1).")

EXIT_OK, EXIT_USER, EXIT_INFRA = 0, 1, 2
CONTRACT_VERSION = "v1"

SCAFFOLD_DIRS = [
    "admin", "archive", "companies", "concepts", "handoffs",
    "ideas", "inbox", "meetings", "papers", "people", "personal",
    "projects", "research", "writing",
]


class UsageError(Exception):
    """User error -> exit 1 (irminsul-io v1)."""


# ---------------------------------------------------------------- output helpers

def emit(obj: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    else:
        _human(obj)


def _human(obj: dict) -> None:
    def walk(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                print(f"{prefix}{k}:")
                walk(prefix + "  ", v)
            elif isinstance(v, list):
                print(f"{prefix}{k}: {', '.join(map(str, v))}")
            else:
                print(f"{prefix}{k}: {v}")

    for k, v in obj.items():
        if isinstance(v, dict):
            print(f"{k}:")
            walk("  ", v)
        elif isinstance(v, list):
            print(f"{k}: {', '.join(map(str, v))}")
        else:
            print(f"{k}: {v}")


# ---------------------------------------------------------------- store plumbing

def _store() -> sqlite3.Connection:
    cpath = cfg.config_path()
    if not cpath.exists():
        raise UsageError(f"no config at {cpath} — run `irminsul init --dir <root>` first")
    db_path = cfg.expand(cfg.resolve("db.path"))
    if not Path(db_path).exists():
        raise UsageError(f"no store at {db_path} — run `irminsul init --dir <root>` first")
    conn = db.connect(db_path)
    # Load vec0 BEFORE migrate: migration deltas DROP/CREATE vec0 virtual
    # tables, which needs the extension registered first. Then auto-upgrade
    # the schema in case the file predates the code — idempotent no-op when
    # the store is already current.
    db.load_vec(conn)
    db.migrate(conn)
    return conn


def _read_input(file: Optional[Path]) -> str:
    if file is not None:
        return Path(file).read_text(encoding="utf-8")
    if sys.stdin.isatty():
        raise UsageError("no input: pass --file PATH or pipe content via stdin")
    return sys.stdin.read()


# ---------------------------------------------------------------- command: init

@app.command()
def init(
    dir_path: str = typer.Option(..., "--dir", help="Vault root (repo.path)"),
    as_json: bool = typer.Option(False, "--json", help="JSON-only stdout"),
):
    """Create store + config + vault scaffold. Idempotent, never clobbers."""
    repo = Path(os.path.abspath(os.path.expanduser(dir_path)))

    db_path = Path(cfg.resolve("db.path"))
    if db_path.exists():
        raise UsageError(
            f"already initialized (store at {db_path}). Re-init would clobber — "
            "use a fresh root or delete the store."
        )

    backup_dir = Path(cfg.expand(cfg.resolve("backup.dir")))
    was_empty = not repo.exists() or not any(repo.iterdir())
    dirs_created = _scaffold(repo)
    _write_index_stub(repo, dirs_created)
    git_status = _git_init_if_fresh(repo, was_empty)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    cfg.write(cfg.resolve_all(repo=str(repo)))

    conn = db.connect(db_path)
    if not db.load_vec(conn):
        conn.close()
        raise UsageError("sqlite-vec failed to load — cannot create the vec0 store")
    try:
        version = db.migrate(conn)
    finally:
        conn.close()

    emit({
        "ok": True,
        "repo_path": str(repo),
        "db_path": str(db_path),
        "schema_version": version,
        "config": str(cfg.config_path()),
        "dirs_created": dirs_created,
        "git": git_status,
    }, as_json)


# ---------------------------------------------------------------- command: doctor

@app.command()
def doctor(as_json: bool = typer.Option(False, "--json")):
    """Health check: structural integrity (config, schema v, fts5/vec0, dup
    slugs). stale_embeds/stale_fts are informational readiness counts."""
    cpath = cfg.config_path()
    if not cpath.exists():
        emit({"ok": False, "checks": {"config": False}, "error": f"no config at {cpath}"}, as_json)
        raise SystemExit(EXIT_USER)

    checks = {"config": True}
    repo = cfg.expand(cfg.resolve("repo.path"))
    db_path = cfg.expand(cfg.resolve("db.path"))
    checks["repo_path_exists"] = Path(repo).exists() if repo else False
    checks["db_path_exists"] = Path(db_path).exists()

    if Path(db_path).exists():
        try:
            conn = db.connect(db_path)
            checks["vec0_loaded"] = db.load_vec(conn)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            checks["schema_version"] = version
            checks["schema_ok"] = version == db.SCHEMA_VERSION
            checks["fts5"] = db.fts_ok(conn)
            checks["vec0_table"] = _table_exists(conn, "chunk_embeddings")
            checks["dup_slugs"] = conn.execute(
                "SELECT count(*) FROM (SELECT slug FROM pages GROUP BY slug HAVING count(*) > 1)"
            ).fetchone()[0]
            if _table_exists(conn, "chunks") and _table_exists(conn, "chunk_embeddings"):
                checks["stale_embeds"] = conn.execute(
                    "SELECT count(*) FROM chunks c LEFT JOIN chunk_embeddings e"
                    " ON e.chunk_id = c.id WHERE e.chunk_id IS NULL"
                ).fetchone()[0]
            else:
                checks["stale_embeds"] = "n/a"
            # FTS index is populated in Phase 2 (triggers + rebuild); until then
            # keyword search is silently empty — surface that as a number.
            checks["stale_fts"] = conn.execute("SELECT count(*) FROM chunks").fetchone()[0] \
                if not _fts_in_sync(conn) else 0
            conn.close()
        except sqlite3.Error as e:
            checks["open_error"] = str(e)

    ok = (
        "open_error" not in checks
        and checks.get("config") is True
        and checks.get("repo_path_exists") is True
        and checks.get("db_path_exists") is True
        and checks.get("vec0_loaded") is True
        and checks.get("schema_ok") is True
        and checks.get("fts5") is True
        and checks.get("vec0_table") is True
        and checks.get("dup_slugs", 1) == 0
        # ok = STRUCTURAL INTEGRITY + no corruption. stale_embeds / stale_fts
        # are readiness counts (informational) — an unembedded/unindexed store
        # is incomplete, not broken; agents read the counts to decide whether
        # retrieval is safe. Fail closed on corruption, not on pipeline lag.
    )
    emit({"ok": ok, "checks": checks}, as_json)
    if not ok:
        raise SystemExit(EXIT_USER)


# ---------------------------------------------------------------- command: put/get/list/stats

@app.command()
def put(
    slug: str = typer.Argument(...),
    file: Optional[Path] = typer.Option(None, "--file", help="read content from file (else stdin)"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Upsert a page by slug (re-chunk; no embedding until Phase 2)."""
    content = _read_input(file)
    meta, body = kbio.parse_frontmatter(content)
    allowed = cfg.resolve("namespace.allow")
    try:
        pages.validate_slug(slug)
        pages.check_namespace(slug, allowed)
    except pages.SlugError as e:
        raise UsageError(str(e))

    with _store() as conn:
        existed = pages.get_page(conn, slug, include_deleted=True) is not None
        page_id = pages.upsert_page(
            conn, slug, body,
            type=meta.get("type", "note"),
            title=meta.get("title"),
            created=meta.get("created"),
            tags=meta.get("tags") or [],
            chunk_size=int(cfg.resolve("chunk.size")),
            chunk_overlap=int(cfg.resolve("chunk.overlap")),
        )
        row = pages.get_page(conn, slug, include_deleted=True)
        n_chunks = conn.execute("SELECT count(*) FROM chunks WHERE page_id = ?", (page_id,)).fetchone()[0]
    emit({
        "ok": True, "slug": slug, "page_id": page_id,
        "changed": "updated" if existed else "created",
        "chunks": n_chunks,
        "type": row["type"], "title": row["title"],
        "created": row["created"], "updated": row["updated"],
    }, as_json)


@app.command()
def get(slug: str = typer.Argument(...), as_json: bool = typer.Option(False, "--json")):
    """Read one page."""
    with _store() as conn:
        row = pages.get_page(conn, slug)
    if row is None:
        raise UsageError(f"page not found: {slug}")
    obj = dict(row)
    obj["content"] = obj.pop("content_md", "")
    if as_json:
        emit(obj, True)
    else:
        title = obj.get("title") or obj["slug"]
        print(f"# {title}\n")
        print(f"type: {obj['type']}   created: {obj['created']}   updated: {obj['updated']}\n")
        print(obj.get("content") or "")


@app.command("list")
def list_cmd(
    limit: int = typer.Option(1000, "--limit", help="cap results (stats is the true count)"),
    include_deleted: bool = typer.Option(False, "--include-deleted"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Enumerate pages."""
    with _store() as conn:
        rows = pages.list_pages(conn, limit=limit, include_deleted=include_deleted)
    items = [
        {"slug": r["slug"], "type": r["type"], "title": r["title"],
         "updated": r["updated"], "deleted_at": r["deleted_at"]}
        for r in rows
    ]
    emit({"items": items, "count": len(items)}, as_json)


@app.command()
def stats(as_json: bool = typer.Option(False, "--json")):
    """Counts: pages/active/deleted/chunks/tags (the true count)."""
    with _store() as conn:
        s = pages.stats(conn)
    emit({"ok": True, **s}, as_json)


# ---------------------------------------------------------------- command: delete/restore/prune

@app.command()
def delete(slug: str = typer.Argument(...), as_json: bool = typer.Option(False, "--json")):
    """Soft-delete a page (recoverable within the window)."""
    try:
        with _store() as conn:
            res = pages.soft_delete(conn, slug)
    except KeyError:
        raise UsageError(f"page not found: {slug}")
    emit({"ok": True, **res}, as_json)


@app.command()
def restore(slug: str = typer.Argument(...), as_json: bool = typer.Option(False, "--json")):
    """Undo a soft delete (page-level)."""
    try:
        with _store() as conn:
            res = pages.restore(conn, slug)
    except KeyError:
        raise UsageError(f"page not found: {slug}")
    emit({"ok": True, **res}, as_json)


@app.command()
def prune(
    older_than: float = typer.Option(
        None, "--older-than", help="hours; default from lifecycle.soft_delete_hours"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Permanently purge soft-deleted pages past the window. Explicit only."""
    hours = older_than if older_than is not None else cfg.resolve("lifecycle.soft_delete_hours")
    with _store() as conn:
        n = pages.prune(conn, hours)
    emit({"ok": True, "pruned": n, "older_than_hours": hours}, as_json)


# ---------------------------------------------------------------- command: backup/recover

@app.command("backup")
def backup_cmd(
    keep: Optional[int] = typer.Option(None, "--keep", help="rotation count; default backup.keep"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Consistent snapshot (VACUUM INTO) to backup.dir, rotated to last N."""
    keep = keep if keep is not None else int(cfg.resolve("backup.keep"))
    db_path = cfg.expand(cfg.resolve("db.path"))
    bdir = cfg.expand(cfg.resolve("backup.dir"))
    res = backup.backup(db_path, bdir, keep=keep)
    emit({"ok": True, **res}, as_json)


@app.command()
def recover(
    target: str = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", help="confirm destructive whole-DB replace"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Replace the whole store from a snapshot. DESTRUCTIVE — requires --yes."""
    if not yes:
        raise UsageError("destructive: pass --yes to replace the store from a snapshot")
    db_path = cfg.expand(cfg.resolve("db.path"))
    bdir = cfg.expand(cfg.resolve("backup.dir"))
    try:
        snap = backup.resolve_snapshot(bdir, target)
    except FileNotFoundError as e:
        raise UsageError(str(e))
    res = backup.recover(db_path, snap)
    emit(res, as_json)


# ---------------------------------------------------------------- command: import/export

@app.command("import")
def import_cmd(
    dir: str = typer.Argument(...),
    dry_run: bool = typer.Option(False, "--dry-run", help="validate only, write nothing"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Batch slug-upsert ingest of a markdown tree. IDEMPOTENT — re-runs safe."""
    allowed = cfg.resolve("namespace.allow")
    if not Path(dir).is_dir():
        raise UsageError(f"not a directory: {dir}")
    with _store() as conn:
        res = kbio.import_dir(conn, dir, allowed=allowed, dry_run=dry_run)
    emit({"ok": True, "dry_run": dry_run, **res}, as_json)


@app.command()
def export(
    dir: str = typer.Option(..., "--dir", help="destination dir for the markdown tree"),
    as_json: bool = typer.Option(False, "--json"),
):
    """DB -> markdown (human/Obsidian view); tree derived from slug prefixes."""
    with _store() as conn:
        res = kbio.export_dir(conn, dir)
    emit({"ok": True, **res}, as_json)


# ---------------------------------------------------------------- command: help/schema

@app.command("help")
def help_cmd(
    cmd: Optional[str] = typer.Argument(None, help="one command's spec"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Self-describing guide — runs ROOTLESS (no Irminsul needed)."""
    specs = _specs()
    if cmd:
        if cmd not in specs:
            raise UsageError(f"unknown command: {cmd}")
        emit(specs[cmd], as_json)
    else:
        emit({
            "contract": "irminsul-io",
            "version": CONTRACT_VERSION,
            "rootless": ["help", "schema", "config"],
            "exit_codes": {"0": "ok", "1": "user error", "2": "infra error"},
            "commands": {k: {"args": v.get("args", []), "status": v["status"]}
                         for k, v in specs.items()},
        }, as_json)


@app.command()
def schema(cmd: str = typer.Argument(...), as_json: bool = typer.Option(False, "--json")):
    """One command's spec: args, JSON output shape, exit codes. ROOTLESS."""
    specs = _specs()
    if cmd not in specs:
        raise UsageError(f"unknown command: {cmd}")
    emit(specs[cmd], as_json)


# ---------------------------------------------------------------- command: config

@app.command("config")
def config_cmd(
    key: Optional[str] = typer.Argument(None, help="knob name (dot path)"),
    value: Optional[str] = typer.Argument(None, help="value to set"),
    verbose: bool = typer.Option(False, "--verbose", help="show source layer"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Read/write config knobs in ~/.irminsul/config.toml (ROOTLESS)."""
    if value is not None:
        _config_set(key, value, as_json)
    elif key is not None:
        _config_get_one(key, verbose, as_json)
    else:
        _config_get_all(as_json)


def _config_get_one(key: str, verbose: bool, as_json: bool) -> None:
    if key not in cfg.DEFAULTS:
        raise UsageError(f"unknown knob: {key} (known: {', '.join(sorted(cfg.DEFAULTS))})")
    value = cfg.resolve(key)
    if verbose:
        emit({"key": key, "value": value, "source": _config_source(key)}, as_json)
    else:
        emit({key: value}, as_json)


def _config_get_all(as_json: bool) -> None:
    out = {k: cfg.resolve(k) for k in sorted(cfg.DEFAULTS)}
    if as_json:
        emit(out, True)
        return
    for k, v in out.items():
        print(f"{k} = {v}")


def _config_source(key: str) -> str:
    if cfg.env_name(key) in os.environ:
        return "env"
    if key in cfg.read():
        return "config"
    return "default"


def _config_set(key, value, as_json: bool) -> None:
    if key not in cfg.DEFAULTS:
        raise UsageError(f"unknown knob: {key}")
    try:
        coerced = cfg.coerce(value, cfg.DEFAULTS[key], key)
    except cfg.ConfigError as e:
        raise UsageError(str(e))
    data = cfg.read()
    data[key] = coerced
    cfg.write(data)
    emit({"key": key, "value": coerced, "written_to": str(cfg.config_path())}, as_json)


# ---------------------------------------------------------------- scaffold helpers

def _scaffold(repo: Path) -> list:
    created = []
    for name in SCAFFOLD_DIRS:
        d = repo / name
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(name)
    return created


def _write_index_stub(repo: Path, created: list) -> None:
    idx = repo / "index.md"
    if not idx.exists():
        idx.write_text("# index\n\n(disk-only hub — content pages live in the store)\n", encoding="utf-8")
        created.append("index.md")


def _git_init_if_fresh(repo: Path, was_empty: bool) -> str:
    if (repo / ".git").exists():
        return "already a git repo (unchanged)"
    if not was_empty:
        return "skipped (existing content)"
    try:
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
        return "initialized"
    except Exception:
        return "failed (best-effort)"


def _table_exists(conn, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master"
        " WHERE type IN ('table', 'virtual table') AND name = ?", (name,)
    ).fetchone() is not None


def _fts_in_sync(conn) -> bool:
    """True when chunks_fts has indexed every chunk (Phase 2 keeps this true via
    triggers; in Phase 1 the index is never populated -> always stale).

    count(*) on an external-content FTS5 table falls through to the content
    table, so it can't detect staleness — chunks_fts_docsize holds one row per
    INDEXED doc, which is the honest count.
    """
    try:
        fts = conn.execute("SELECT count(*) FROM chunks_fts_docsize").fetchone()[0]
        return fts == conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    except sqlite3.Error:
        return False


# ---------------------------------------------------------------- spec table (irminsul-io v1)

def _specs() -> dict:
    codes = {"0": "ok", "1": "user error", "2": "infra error"}
    return {
        "init": {"args": ["--dir PATH (required)", "--json"],
                 "output": {"ok": True, "repo_path": "str", "db_path": "str", "schema_version": 2,
                            "config": "path", "dirs_created": ["str"], "git": "str"},
                 "exit_codes": codes, "status": "implemented", "rootless": False},
        "doctor": {"args": ["--json"],
                   "output": {"ok": "bool", "checks": {"config": "bool", "repo_path_exists": "bool",
                                                       "db_path_exists": "bool", "vec0_loaded": "bool",
                                                       "schema_version": "int", "schema_ok": "bool",
                                                       "fts5": "bool", "vec0_table": "bool",
                                                       "dup_slugs": "int", "stale_embeds": "int",
                                                       "stale_fts": "int"}},
                   "exit_codes": codes, "status": "implemented", "rootless": False},
        "put": {"args": ["<slug>", "--file PATH | stdin", "--json"],
                "output": {"ok": True, "slug": "str", "page_id": "int", "changed": "created|updated",
                           "chunks": "int", "type": "str", "title": "str|None",
                           "created": "str", "updated": "str"},
                "exit_codes": codes, "status": "implemented", "rootless": False},
        "get": {"args": ["<slug>", "--json"],
                "output": {"slug": "str", "type": "str", "title": "str", "created": "str",
                           "updated": "str", "deleted_at": "str|None", "content": "str"},
                "exit_codes": codes, "status": "implemented", "rootless": False},
        "list": {"args": ["--limit N", "--include-deleted", "--json"],
                 "output": {"items": [{"slug", "type", "title", "updated", "deleted_at"}], "count": "int"},
                 "exit_codes": codes, "status": "implemented", "rootless": False},
        "stats": {"args": ["--json"],
                  "output": {"ok": True, "pages": "int", "active": "int", "deleted": "int",
                             "chunks": "int", "tags": "int"},
                  "exit_codes": codes, "status": "implemented", "rootless": False},
        "delete": {"args": ["<slug>", "--json"],
                   "output": {"ok": True, "slug": "str", "deleted_at": "str"},
                   "exit_codes": codes, "status": "implemented", "rootless": False},
        "restore": {"args": ["<slug>", "--json"],
                    "output": {"ok": True, "slug": "str", "deleted_at": "None"},
                    "exit_codes": codes, "status": "implemented", "rootless": False},
        "prune": {"args": ["--older-than HOURS", "--json"],
                  "output": {"ok": True, "pruned": "int", "older_than_hours": "float"},
                  "exit_codes": codes, "status": "implemented", "rootless": False},
        "backup": {"args": ["--keep N", "--json"],
                   "output": {"ok": True, "snapshot": "path", "size": "int", "kept": "int",
                              "removed": ["str"]},
                   "exit_codes": codes, "status": "implemented", "rootless": False},
        "recover": {"args": ["<ts|path>", "--yes", "--json"],
                    "output": {"ok": True, "from": "path", "to": "path"},
                    "exit_codes": codes, "status": "implemented", "rootless": False},
        "import": {"args": ["<dir>", "--dry-run", "--json"],
                   "output": {"ok": True, "dry_run": "bool", "imported": "int",
                              "skipped": "int", "errors": [{"file", "error"}]},
                   "exit_codes": codes, "status": "implemented", "rootless": False},
        "export": {"args": ["--dir PATH (required)", "--json"],
                   "output": {"ok": True, "exported": "int", "dir": "str"},
                   "exit_codes": codes, "status": "implemented", "rootless": False},
        "help": {"args": ["[cmd]", "--json"],
                 "output": {"contract": "irminsul-io", "version": "v1", "commands": "{name: {args, status}}"},
                 "exit_codes": codes, "status": "implemented", "rootless": True},
        "schema": {"args": ["<cmd>", "--json"],
                   "output": "one command's spec (args, json shape, exit codes)",
                   "exit_codes": codes, "status": "implemented", "rootless": True},
        "config": {"args": ["[key]", "[value]", "--json", "--verbose"],
                   "output": {"key": "str", "value": "scalar", "source": "cli|env|config|default"},
                   "exit_codes": codes, "status": "implemented", "rootless": True},
        "embed": {"args": ["--stale", "--json"], "status": "planned (Phase 2)", "rootless": False},
        "graph": {"args": ["<slug>", "--depth N"], "status": "planned (Phase 3)", "rootless": False},
        "rag": {"args": ["<question>", "--k N", "--json"], "status": "planned (post-v1 Phase 4)",
                "rootless": False},
    }


def main():
    try:
        app()
    except click.exceptions.UsageError as e:
        print(f"irminsul: {e.format_message()}", file=sys.stderr)
        raise SystemExit(EXIT_USER)
    except UsageError as e:
        print(f"irminsul: {e}", file=sys.stderr)
        raise SystemExit(EXIT_USER)
    except KeyError as e:
        print(f"irminsul: not found: {e}", file=sys.stderr)
        raise SystemExit(EXIT_USER)
    except sqlite3.Error as e:
        print(f"irminsul: store error: {e}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:  # pragma: no cover
        print(f"irminsul: unexpected error: {e}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)


if __name__ == "__main__":
    main()
