"""Phase 2 — schema v1->v2 migration, FTS trigger sync, vec0 writes, embed seam."""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from irminsul import db, embed, vectors

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_V1 = """
PRAGMA user_version = 1;
create table pages (
  id integer primary key,
  slug text unique not null,
  type text not null default 'note',
  title text,
  created text not null default (datetime('now')),
  updated text not null default (datetime('now')),
  deleted_at text,
  content_md text not null
);
create table chunks (
  id integer primary key,
  page_id integer not null references pages(id),
  seq integer not null,
  chunk_text text not null,
  embed_model text
);
create index idx_chunks_page on chunks(page_id);
create virtual table chunks_fts using fts5(chunk_text, content='chunks', content_rowid='id');
create virtual table chunk_embeddings using vec0(chunk_id integer primary key, embedding float[1280]);
create table links (src_chunk_id integer, dst_slug text, kind text default 'wikilink');
create table tags (page_id integer references pages(id), tag text);
create table meta (k text primary key, v text);
insert into meta(k, v) values ('schema_version', '1'), ('embed_model', 'zembed-1');
"""


def test_migrate_v1_to_v2_preserves_data_and_lives_fts(tmp_path):
    p = tmp_path / "irminsul.db"
    conn = db.connect(p)
    assert db.load_vec(conn)
    conn.executescript(SCHEMA_V1)
    # seed a v1 page + chunk (FTS index absent — pre-trigger world)
    conn.execute("INSERT INTO pages(slug, content_md) VALUES ('concepts/old', 'old body')")
    conn.execute("INSERT INTO chunks(page_id, seq, chunk_text) VALUES (1, 0, 'legacy chunk text')")
    conn.commit()

    v = db.migrate(conn)
    assert v == 2
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2

    # vec0 recreated at 1024
    sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'chunk_embeddings'").fetchone()[0]
    assert "float[1024]" in sql and "float[1280]" not in sql

    # FTS backfilled by the migration's 'rebuild' + triggers keep it live
    fts = conn.execute("SELECT count(*) FROM chunks_fts_docsize").fetchone()[0]
    assert fts == conn.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1
    hits = conn.execute(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'legacy'").fetchall()
    assert [r[0] for r in hits] == [1]

    # meta stamped to the new model
    meta = {r["k"]: r["v"] for r in conn.execute("SELECT k, v FROM meta")}
    assert meta["schema_version"] == "2" and meta["embed_model"] == "voyage-4-large"
    conn.close()


def test_fts_triggers_stay_in_sync_on_write(tmp_path):
    p = tmp_path / "irminsul.db"
    conn = db.connect(p)
    assert db.load_vec(conn)
    db.migrate(conn)
    conn.execute("INSERT INTO pages(slug, content_md) VALUES ('a/b', 'x')")
    ch = conn.execute(
        "INSERT INTO chunks(page_id, seq, chunk_text) VALUES (1, 0, 'needle in haystack')"
    )
    assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 1
    # delete the chunk -> FTS row disappears (external-content 'delete' trigger)
    conn.execute("DELETE FROM chunks WHERE id = ?", (ch.lastrowid,))
    assert conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] == 0
    conn.close()


def test_vectors_add_embeddings_and_stale(tmp_path):
    p = tmp_path / "irminsul.db"
    conn = db.connect(p)
    assert db.load_vec(conn)
    db.migrate(conn)
    conn.execute("INSERT INTO pages(slug, content_md) VALUES ('a/b', 'x')")
    conn.execute("INSERT INTO chunks(page_id, seq, chunk_text) VALUES (1, 0, 'one')")
    conn.execute("INSERT INTO chunks(page_id, seq, chunk_text) VALUES (1, 1, 'two')")
    conn.commit()

    fake = embed.FakeEmbedder(dim=1024)
    cids = [1, 2]
    n = vectors.add_embeddings(conn, cids, fake.embed(["one", "two"]), "voyage-4-large")
    assert n == 2
    assert conn.execute("SELECT count(*) FROM chunk_embeddings").fetchone()[0] == 2
    assert conn.execute(
        "SELECT embed_model FROM chunks WHERE id = 1").fetchone()[0] == "voyage-4-large"
    assert vectors.stale_chunk_ids(conn, "voyage-4-large") == []

    # model bump -> everything stale again (delete+insert recompute path)
    stale = vectors.stale_chunk_ids(conn, "voyage-4-large-v2")
    assert stale == [1, 2]
    conn.close()


def test_fake_embedder_deterministic_dims():
    fake = embed.FakeEmbedder(dim=1024)
    a = fake.embed(["hello"])
    b = fake.embed(["hello"])
    assert a == b
    assert len(a[0]) == 1024
    assert abs(sum(x * x for x in a[0]) - 1.0) < 1e-6  # L2-normalized
    scores = fake.rerank("hello", ["hello", "zzzz thing wholly unrelated"])
    assert scores[0] > scores[1]  # identical text scores highest (cos=1)


def test_voyage_rerank_uses_configured_model(monkeypatch):
    import httpx

    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"index": 0, "relevance_score": 0.9}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    monkeypatch.setattr(httpx, "post", fake_post)
    v = embed.get_embedder("voyage", dim=8, rerank_model="rerank-2.5-test")
    scores = v.rerank("q", ["doc"])
    assert captured["json"]["model"] == "rerank-2.5-test"  # knob, not hardcode
    assert scores == [0.9]


# ------------------------------------------------------------------ CLI: migrate / doctor --fix

def _cli(args, home, input=None, extra_env=None):
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env.setdefault("IRMINSUL_EMBED_PROVIDER", "fake")  # CLI tests run offline
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "irminsul", *args],
        capture_output=True, text=True, env=env, input=input, cwd=str(ROOT),
    )


def _home_with_v1_store(tmp_path):
    """init (writes config + v2 store), then replace the file with a v1 store."""
    home = tmp_path / "h"
    root = tmp_path / "vault"
    r = _cli(["init", "--dir", str(root), "--json"], home)
    assert r.returncode == 0, r.stderr
    dbp = home / ".irminsul" / "irminsul.db"
    dbp.unlink()
    conn = sqlite3.connect(dbp)
    assert db.load_vec(conn)
    conn.executescript(SCHEMA_V1)
    conn.commit()
    conn.close()
    return home, dbp


def test_migrate_command_upgrades_and_idempotent(tmp_path):
    home, dbp = _home_with_v1_store(tmp_path)
    r = _cli(["migrate", "--json"], home)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["from"] == 1 and out["to"] == 2 and out["upgraded"] is True
    # idempotent: no-op on an already-current store
    r2 = _cli(["migrate", "--json"], home)
    assert r2.returncode == 0, r2.stderr
    out2 = json.loads(r2.stdout)
    assert out2["from"] == 2 and out2["to"] == 2 and out2["upgraded"] is False
    conn = db.connect(dbp)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    conn.close()


def test_doctor_reports_stale_then_fix_heals(tmp_path):
    home, dbp = _home_with_v1_store(tmp_path)
    # report-only: stale schema flagged with a remedy, file left untouched
    r = _cli(["doctor", "--json"], home)
    assert r.returncode == 1  # fail-closed
    out = json.loads(r.stdout)
    assert out["ok"] is False
    assert out["checks"]["schema_version"] == 1 and out["checks"]["schema_ok"] is False
    assert "remedy" in out and "migrate" in out["remedy"]
    conn = db.connect(dbp)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1  # untouched
    conn.close()
    # --fix migrates then re-checks green
    r2 = _cli(["doctor", "--fix", "--json"], home)
    assert r2.returncode == 0, r2.stderr
    out2 = json.loads(r2.stdout)
    assert out2["ok"] is True and out2["migrated"] is True
    assert out2["checks"]["schema_version"] == 2 and out2["checks"]["schema_ok"] is True
    assert "remedy" not in out2
    # second --fix is a no-op
    r3 = _cli(["doctor", "--fix", "--json"], home)
    out3 = json.loads(r3.stdout)
    assert out3["ok"] is True and out3["migrated"] is False


def test_embed_stale_heals_and_is_idempotent(tmp_path):
    env = {"IRMINSUL_EMBED_PROVIDER": "fake"}  # offline, deterministic vectors
    home = tmp_path / "h"
    root = tmp_path / "vault"
    r = _cli(["init", "--dir", str(root), "--json"], home, extra_env=env)
    assert r.returncode == 0, r.stderr
    # put --no-embed leaves fresh chunks stale (embed-on-write is the default)
    r = _cli(["put", "projects/foo", "--no-embed", "--json"], home, input="cat facts here", extra_env=env)
    assert r.returncode == 0, r.stderr
    assert json.loads(_cli(["doctor", "--json"], home, extra_env=env).stdout)["checks"]["stale_embeds"] >= 1

    # repair: embed --stale finds the NULL-tagged chunk and stamps it
    r = _cli(["embed", "--stale", "--json"], home, extra_env=env)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["ok"] is True and out["stale"] is True and out["embedded"] >= 1

    d = json.loads(_cli(["doctor", "--json"], home, extra_env=env).stdout)
    assert d["checks"]["stale_embeds"] == 0  # store fully green

    # idempotent: nothing left stale
    out2 = json.loads(_cli(["embed", "--stale", "--json"], home, extra_env=env).stdout)
    assert out2["embedded"] == 0


# ------------------------------------------------------------------ Phase 2: put embed-on-write + search

def test_put_embed_on_write_default_and_no_embed(tmp_path):
    env = {"IRMINSUL_EMBED_PROVIDER": "fake"}
    home = tmp_path / "h"
    root = tmp_path / "vault"
    assert _cli(["init", "--dir", str(root), "--json"], home, extra_env=env).returncode == 0
    # default: put embeds its fresh chunks
    r = _cli(["put", "projects/a", "--json"], home, input="auto embedded here", extra_env=env)
    out = json.loads(r.stdout)
    assert out["embedded"] >= 1 and "embed_error" not in out
    assert json.loads(_cli(["doctor", "--json"], home, extra_env=env).stdout)["checks"]["stale_embeds"] == 0
    # --no-embed leaves chunks stale for the repair verb
    _cli(["put", "projects/b", "--no-embed", "--json"], home, input="stale chunk here", extra_env=env)
    outb = json.loads(_cli(["put", "projects/b", "--no-embed", "--json"], home,
                           input="stale chunk here", extra_env=env).stdout)
    assert outb.get("no_embed") is True
    assert json.loads(_cli(["doctor", "--json"], home, extra_env=env).stdout)["checks"]["stale_embeds"] >= 1


def test_search_hybrid_and_excludes_deleted(tmp_path):
    env = {"IRMINSUL_EMBED_PROVIDER": "fake"}
    home = tmp_path / "h"
    root = tmp_path / "vault"
    assert _cli(["init", "--dir", str(root), "--json"], home, extra_env=env).returncode == 0
    _cli(["put", "concepts/cats", "--json"], home, input="cats meow and purr loudly in the kitchen", extra_env=env)
    _cli(["put", "concepts/dogs", "--json"], home, input="dogs bark and wag tails in the park", extra_env=env)
    out = json.loads(_cli(["search", "meow cats", "--k", "3", "--json"], home, extra_env=env).stdout)
    assert out["ok"] is True and out["count"] >= 1
    slugs = {res["slug"] for res in out["results"]}
    assert "concepts/cats" in slugs
    assert all({"slug", "score", "excerpt", "chunk_id"} <= set(res) for res in out["results"])
    # soft-delete a matching page -> must vanish from search
    _cli(["delete", "concepts/cats", "--json"], home, extra_env=env)
    out2 = json.loads(_cli(["search", "meow cats", "--k", "3", "--json"], home, extra_env=env).stdout)
    assert "concepts/cats" not in {res["slug"] for res in out2["results"]}


def test_search_keyword_only_without_embeddings(tmp_path):
    home = tmp_path / "h"
    root = tmp_path / "vault"
    assert _cli(["init", "--dir", str(root), "--json"], home).returncode == 0
    # no vectors at all (--no-embed, nothing ever embedded) -> FTS leg only, no crash
    _cli(["put", "concepts/kw", "--no-embed", "--json"], home, input="needle in the haystack here")
    out = json.loads(_cli(["search", "needle haystack", "--k", "3", "--json"], home).stdout)
    assert out["count"] >= 1 and out["results"][0]["slug"] == "concepts/kw"


def test_search_k_cap_enforced(tmp_path):
    home = tmp_path / "h"
    home.mkdir(parents=True)
    r = _cli(["search", "x", "--k", "25", "--json"], home)
    assert r.returncode == 1  # typer min/max -> user error exit
