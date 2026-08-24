"""kb — Typer CLI command surface (Phase 0: init, doctor, help, schema, config)."""

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

from . import config as cfg
from . import db

app = typer.Typer(no_args_is_help=True, help="kb — agent knowledge base (kb-io v1).")

EXIT_OK, EXIT_USER, EXIT_INFRA = 0, 1, 2
CONTRACT_VERSION = "v1"

SCAFFOLD_DIRS = [
    "admin", "archive", "companies", "concepts", "handoffs",
    "ideas", "inbox", "meetings", "papers", "people", "personal",
    "projects", "research", "writing",
]


class UsageError(Exception):
    """User error -> exit 1 (kb-io v1)."""


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


# ---------------------------------------------------------------- command surface

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


@app.command()
def doctor(as_json: bool = typer.Option(False, "--json")):
    """Health check: config, schema v, fts5/vec0, dup slugs, stale embeds."""
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
        and checks.get("stale_embeds") in (0, "n/a")
    )
    emit({"ok": ok, "checks": checks}, as_json)
    if not ok:
        raise SystemExit(EXIT_USER)


@app.command("help")
def help_cmd(
    cmd: Optional[str] = typer.Argument(None, help="one command's spec"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Self-describing guide — runs ROOTLESS (no KB needed)."""
    specs = _specs()
    if cmd:
        if cmd not in specs:
            raise UsageError(f"unknown command: {cmd}")
        emit(specs[cmd], as_json)
    else:
        emit({
            "contract": "kb-io",
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


@app.command("config")
def config_cmd(
    key: Optional[str] = typer.Argument(None, help="knob name (dot path)"),
    value: Optional[str] = typer.Argument(None, help="value to set"),
    verbose: bool = typer.Option(False, "--verbose", help="show source layer"),
    as_json: bool = typer.Option(False, "--json"),
):
    """Read/write config knobs in ~/.kb/config.toml (ROOTLESS)."""
    if value is not None:
        _config_set(key, value, as_json)
    elif key is not None:
        _config_get_one(key, verbose, as_json)
    else:
        _config_get_all(as_json)


# ---------------------------------------------------------------- config helpers

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


# ---------------------------------------------------------------- spec table (kb-io v1)

def _specs() -> dict:
    codes = {"0": "ok", "1": "user error", "2": "infra error"}
    return {
        "init": {
            "args": ["--dir PATH (required)", "--json"],
            "output": {"ok": True, "repo_path": "str", "db_path": "str", "schema_version": 1,
                       "config": "path", "dirs_created": ["str"], "git": "str"},
            "exit_codes": codes, "status": "implemented", "rootless": False,
        },
        "doctor": {
            "args": ["--json"],
            "output": {"ok": "bool", "checks": {
                "config": "bool", "repo_path_exists": "bool", "db_path_exists": "bool",
                "vec0_loaded": "bool", "schema_version": "int", "schema_ok": "bool",
                "fts5": "bool", "vec0_table": "bool", "dup_slugs": "int", "stale_embeds": "int"}},
            "exit_codes": codes, "status": "implemented", "rootless": False,
        },
        "help": {
            "args": ["[cmd]", "--json"],
            "output": {"contract": "kb-io", "version": "v1", "commands": "{name: {args, status}}"},
            "exit_codes": codes, "status": "implemented", "rootless": True,
        },
        "schema": {
            "args": ["<cmd>", "--json"],
            "output": "one command's spec (args, json shape, exit codes)",
            "exit_codes": codes, "status": "implemented", "rootless": True,
        },
        "config": {
            "args": ["[key]", "[value]", "--json", "--verbose"],
            "output": {"key": "str", "value": "scalar", "source": "cli|env|config|default"},
            "exit_codes": codes, "status": "implemented", "rootless": True,
        },
        "put": {"status": "planned (Phase 1)"},
        "get": {"status": "planned (Phase 1)"},
        "list": {"status": "planned (Phase 1)"},
        "stats": {"status": "planned (Phase 1)"},
        "delete": {"status": "planned (Phase 1)"},
        "restore": {"status": "planned (Phase 1)"},
        "backup": {"status": "planned (Phase 1)"},
        "recover": {"status": "planned (Phase 1)"},
        "import": {"status": "planned (Phase 1)"},
        "export": {"status": "planned (Phase 1)"},
        "embed": {"status": "planned (Phase 2)"},
        "graph": {"status": "planned (Phase 3)"},
        "prune": {"status": "planned (Phase 3)"},
        "rag": {"status": "planned (post-v1 Phase 4)"},
    }


def main():
    try:
        app()
    except click.exceptions.UsageError as e:
        print(f"kb: {e.format_message()}", file=sys.stderr)
        raise SystemExit(EXIT_USER)
    except UsageError as e:
        print(f"kb: {e}", file=sys.stderr)
        raise SystemExit(EXIT_USER)
    except sqlite3.Error as e:
        print(f"kb: store error: {e}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as e:  # pragma: no cover
        print(f"kb: unexpected error: {e}", file=sys.stderr)
        raise SystemExit(EXIT_INFRA)


if __name__ == "__main__":
    main()
