"""~/.irminsul/config.toml bootstrap config — the pointer that lives OUTSIDE the DB.

Why a file, not a DB row: every command's first step is resolving the store,
and the pointer must terminate outside the DB (a pointer stored inside the
thing it points at is circular).

Resolution order everywhere: CLI flag > env var (IRMINSUL_<KEY>) > config file > built-in default.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

APP_DIR_NAME = ".irminsul"

DEFAULTS: "dict[str, object]" = {
    "repo.path": None,  # required — human vault root (export/import/Obsidian)
    "db.path": "~/.irminsul/irminsul.db",  # the SQLite store (kept outside the vault, gbrain-style)
    "backup.dir": "~/.irminsul/backups",
    "backup.keep": 10,
    "namespace.allow": [
        "archive", "companies", "concepts", "ideas", "inbox", "meetings",
        "papers", "people", "personal", "projects", "research", "writing",
    ],  # content namespaces only; admin/handoffs are disk-only zones
    "chunk.size": 600,
    "chunk.overlap": 60,
    "search.limit": 5,
    "search.rrf_k": 60,
    "search.w_fs": 1.0,
    "search.w_vec": 1.0,
    "search.rerank": True,
    "lifecycle.soft_delete_hours": 72,
}

PATH_KEYS = {"repo.path", "db.path", "backup.dir"}


class ConfigError(ValueError):
    pass


def app_dir() -> Path:
    return Path.home() / APP_DIR_NAME


def config_path() -> Path:
    return app_dir() / "config.toml"


def expand(value):
    if isinstance(value, str):
        return os.path.abspath(os.path.expanduser(value))
    return value


def read() -> dict:
    p = config_path()
    if not p.exists():
        return {}
    with open(p, "rb") as f:
        return tomllib.load(f)


def resolve(key, cli_value=None, use_env=True):
    """CLI flag > env (IRMINSUL_<KEY>) > config file > built-in default."""
    if cli_value is not None:
        return _expand_if_path(key, cli_value)
    if use_env:
        env_value = os.environ.get(env_name(key))
        if env_value is not None:
            return _expand_if_path(key, _coerce(env_value, DEFAULTS.get(key)))
    cfg = read()
    if key in cfg and cfg[key] is not None:
        return _expand_if_path(key, cfg[key])
    return _expand_if_path(key, DEFAULTS.get(key))


def resolve_all(repo=None) -> dict:
    """Full snapshot for writing a self-documenting config.toml at init."""
    data = dict(DEFAULTS)
    if repo is not None:
        data["repo.path"] = str(repo)
    return {k: _expand_if_path(k, v) for k, v in data.items() if v is not None}


def env_name(key: str) -> str:
    return "IRMINSUL_" + key.upper().replace(".", "_").replace("-", "_")


def coerce(value, default, key=None):
    try:
        if isinstance(default, bool):
            return str(value).strip().lower() in ("1", "true", "yes", "on")
        if isinstance(default, int) and not isinstance(default, bool):
            return int(str(value).strip())
        if isinstance(default, float) and not isinstance(default, bool):
            return float(str(value).strip())
        if isinstance(default, list):
            if isinstance(value, str):
                return [x.strip() for x in value.split(",") if x.strip()]
            return list(value)
        return str(value)
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"{key or 'value'}: cannot interpret {value!r} as {type(default).__name__}"
        ) from e


def _coerce(env_value, default):
    try:
        return coerce(env_value, default)
    except ConfigError:
        return env_value


def _expand_if_path(key, value):
    return expand(value) if key in PATH_KEYS else value


def write(values: dict) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(_toml_dump(values))


def _toml_dump(values: dict) -> str:
    return "\n".join(_kv(k, v) for k, v in sorted(values.items()) if v is not None) + "\n"


def _kv(key, value) -> str:
    # Quoted key: dots are literal, so "repo.path" stays ONE key (no table nesting).
    return f"{_quote_key(key)} = {_vtoml(value)}"


def _quote_key(key: str) -> str:
    return f'"{key}"'


def _vtoml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_vtoml(x) for x in value) + "]"
    s = str(value)
    # Windows paths contain backslashes -> TOML literal strings (no escaping).
    if "\\" in s or "'" in s:
        return "'" + s.replace("'", "''") + "'"
    return '"' + s.replace('"', '\\"') + '"'
