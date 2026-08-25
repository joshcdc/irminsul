import json
import os
import subprocess
import sys
from pathlib import Path

from irminsul import config as cfg
from irminsul import db

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_config_roundtrip_with_backslash_and_space_paths(tmp_path, monkeypatch):
    homedir = tmp_path / "home with space"
    monkeypatch.setenv("HOME", str(homedir))
    monkeypatch.setenv("USERPROFILE", str(homedir))
    data = {
        "repo.path": str(homedir / "my vault"),
        "db.path": str(homedir / ".irminsul" / "irminsul.db"),
        "backup.keep": 10,
        "search.rerank": True,
        "namespace.allow": ["concepts", "projects"],
    }
    cfg.write(data)
    back = cfg.read()
    assert back["repo.path"] == str(homedir / "my vault")
    assert back["db.path"] == str(homedir / ".irminsul" / "irminsul.db")
    assert back["backup.keep"] == 10
    assert back["search.rerank"] is True
    assert back["namespace.allow"] == ["concepts", "projects"]
    # resolve() expands paths and falls through layers
    assert cfg.resolve("repo.path", use_env=False) == str(homedir / "my vault")
    assert cfg.resolve("backup.keep", use_env=False) == 10


def test_config_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "h"))
    monkeypatch.setenv("IRMINSUL_BACKUP_KEEP", "5")
    assert cfg.resolve("backup.keep") == 5
    assert cfg.resolve("repo.path") is None  # no config, no default


def test_migrate_idempotent(tmp_path):
    p = tmp_path / "irminsul.db"
    conn = db.connect(p)
    assert db.load_vec(conn), "sqlite-vec must load in this env"
    assert db.fts_ok(conn) is False or True  # table doesn't exist yet; just must not raise
    v = db.migrate(conn)
    assert v == db.SCHEMA_VERSION
    assert db.fts_ok(conn) is True
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE name IN ('chunks_fts', 'chunk_embeddings', 'pages', 'meta')"
    )}
    assert {"chunks_fts", "chunk_embeddings", "pages", "meta"} <= tables
    # re-run migrate = no-op, no error
    assert db.migrate(conn) == db.SCHEMA_VERSION
    meta = {r["k"]: r["v"] for r in conn.execute("SELECT k, v FROM meta")}
    assert meta["schema_version"] == "2"
    assert meta["embed_model"] == "voyage-4-large"
    conn.close()


def _cli(*args, cwd=PROJECT_ROOT, home):
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", "irminsul", *args],
        capture_output=True, text=True, env=env, cwd=str(cwd),
    )


def test_cli_help_rootless(tmp_path):
    res = _cli("help", "--json", home=tmp_path / "h")
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    assert data["contract"] == "irminsul-io"
    assert data["version"] == "v1"
    assert "init" in data["commands"]


def test_cli_schema_rootless(tmp_path):
    res = _cli("schema", "doctor", "--json", home=tmp_path / "h")
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout)["status"] == "implemented"


def test_cli_init_doctor_smoke(tmp_path):
    home = tmp_path / "h"
    root = tmp_path / "vault"
    res = _cli("init", "--dir", str(root), "--json", home=home)
    assert res.returncode == 0, res.stderr
    init = json.loads(res.stdout)
    assert init["ok"] is True
    assert init["repo_path"] == str(root)
    assert (home / ".irminsul" / "config.toml").exists()
    assert init["dirs_created"]  # scaffold happened
    assert "concepts" in init["dirs_created"]
    assert init["git"] in ("initialized", "failed (best-effort)")  # fresh root -> git init

    res2 = _cli("doctor", "--json", home=home)
    assert res2.returncode == 0, res2.stderr
    doc = json.loads(res2.stdout)
    assert doc["ok"] is True
    assert doc["checks"]["schema_ok"] is True
    assert doc["checks"]["fts5"] is True
    assert doc["checks"]["vec0_table"] is True
    assert doc["checks"]["vec0_loaded"] is True

    # re-init on existing store must fail closed (exit 1, stderr)
    res3 = _cli("init", "--dir", str(root), "--json", home=home)
    assert res3.returncode == 1
    assert "already initialized" in res3.stderr
