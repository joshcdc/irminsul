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
