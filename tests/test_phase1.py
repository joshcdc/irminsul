import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from irminsul import chunk as chunking
from irminsul import db
from irminsul import io as kbio
from irminsul import pages

ROOT = Path(__file__).resolve().parent.parent


def _cli(args, home, input=None):
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env.setdefault("IRMINSUL_EMBED_PROVIDER", "fake")  # CLI tests run offline
    return subprocess.run(
        [sys.executable, "-m", "irminsul", *args],
        capture_output=True, text=True, env=env, input=input, cwd=str(ROOT),
    )


def _setup(tmp_path):
    home = tmp_path / "h"
    root = tmp_path / "vault"
    r = _cli(["init", "--dir", str(root), "--json"], home)
    assert r.returncode == 0, r.stderr
    return home, root


# ------------------------------------------------------------------ unit: chunking

def test_chunk_basic():
    text = " ".join("word" for _ in range(400))
    parts = chunking.chunk_text(text, size=120, overlap=30)
    assert parts and all(parts)
    for p in parts:
        assert len(p) <= 160  # size + backoff bound
    assert len(parts) > 1


def test_chunk_small_and_empty():
    assert chunking.chunk_text("hi", 600, 60) == ["hi"]
    assert chunking.chunk_text("", 600, 60) == []


# ------------------------------------------------------------------ unit: frontmatter

def test_frontmatter_parse():
    text = ("---\ntype: project\ntitle: Foo Bar\ncreated: 2026-01-01 00:00:00\ntags:\n"
            "  - a\n  - b\n---\n\nbody text\n\nmore")
    meta, body = kbio.parse_frontmatter(text)
    assert meta["type"] == "project"
    assert meta["title"] == "Foo Bar"
    assert meta["created"] == "2026-01-01 00:00:00"
    assert meta["tags"] == ["a", "b"]
    assert body == "body text\n\nmore"


def test_frontmatter_absent():
    assert kbio.parse_frontmatter("plain body") == ({}, "plain body")


def test_frontmatter_build_roundtrip():
    fm = kbio.build_frontmatter(type="note", title="X", created="c", updated="u", tags=["t"])
    meta, body = kbio.parse_frontmatter(fm + "BODY")
    assert meta["type"] == "note" and meta["title"] == "X"
    assert meta["tags"] == ["t"]
    assert body == "BODY"


# ------------------------------------------------------------------ unit: slugs

def test_slug_validation():
    pages.validate_slug("concepts/foo_bar-1")
    pages.validate_slug("a")
    for bad in ("../x", "concepts/..", "concepts//x", "Concepts/x", "concepts/foo bar", ""):
        with pytest.raises(pages.SlugError):
            pages.validate_slug(bad)


def test_namespace_allowlist():
    allowed = ["concepts", "projects"]
    assert pages.check_namespace("concepts/x", allowed) == "concepts/x"
    with pytest.raises(pages.SlugError):
        pages.check_namespace("admin/x", allowed)


def test_tags_normalized_and_deduped(tmp_path):
    home, root = _setup(tmp_path)
    fm = "---\ntype: note\ntags:\n  - Alpha\n  - alpha\n  - '  Beta  '\n---\n\nbody"
    r = _cli(["put", "concepts/tags", "--json"], home, input=fm)
    assert r.returncode == 0, r.stderr
    s = json.loads(_cli(["stats", "--json"], home).stdout)
    assert s["tags"] == 2  # deduped Alpha/alpha -> one row; 'Beta' stripped


# ------------------------------------------------------------------ integration: put/list/stats

def test_put_upsert_stats_and_fail_closed(tmp_path):
    home, root = _setup(tmp_path)

    r = _cli(["put", "concepts/alpha", "--json"], home, input="# Alpha\n\nhello world")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["ok"] and d["changed"] == "created" and d["chunks"] == 1

    r2 = _cli(["put", "concepts/alpha", "--json"], home, input="# Alpha v2")
    assert json.loads(r2.stdout)["changed"] == "updated"

    s = json.loads(_cli(["stats", "--json"], home).stdout)
    assert s["pages"] == 1 and s["active"] == 1 and s["chunks"] == 1  # delta 0 on re-put

    g = json.loads(_cli(["get", "concepts/alpha", "--json"], home).stdout)
    assert g["content"] == "# Alpha v2"

    # fail closed: traversal slug + disallowed namespace
    r3 = _cli(["put", "../../x", "--json"], home, input="x")
    assert r3.returncode == 1
    r4 = _cli(["put", "admin/x", "--json"], home, input="x")
    assert r4.returncode == 1 and "allowlist" in r4.stderr

    lst = json.loads(_cli(["list", "--json"], home).stdout)
    assert lst["count"] == 1 and lst["items"][0]["slug"] == "concepts/alpha"


# ------------------------------------------------------------------ integration: delete/restore/prune

def test_delete_restore_prune(tmp_path):
    home, root = _setup(tmp_path)
    _cli(["put", "concepts/gone", "--json"], home, input="bye")

    delr = json.loads(_cli(["delete", "concepts/gone", "--json"], home).stdout)
    assert delr["deleted_at"]

    assert json.loads(_cli(["list", "--json"], home).stdout)["count"] == 0  # hidden
    assert json.loads(_cli(["list", "--include-deleted", "--json"], home).stdout)["count"] == 1

    rr = _cli(["restore", "concepts/gone", "--json"], home)
    assert rr.returncode == 0, rr.stderr
    assert json.loads(rr.stdout)["deleted_at"] is None
    assert json.loads(_cli(["list", "--json"], home).stdout)["count"] == 1

    _cli(["delete", "concepts/gone", "--json"], home)
    pr = json.loads(_cli(["prune", "--older-than", "0", "--json"], home).stdout)
    assert pr["pruned"] == 1
    assert json.loads(_cli(["list", "--include-deleted", "--json"], home).stdout)["count"] == 0
    assert json.loads(_cli(["stats", "--json"], home).stdout)["pages"] == 0  # hard-deleted


# ------------------------------------------------------------------ integration: export/import round-trip + idempotence

def test_export_import_roundtrip_idempotent(tmp_path):
    home, root = _setup(tmp_path)
    _cli(["put", "concepts/a", "--json"], home, input="# A\naaa")
    _cli(["put", "projects/b", "--json"], home, input="# B\nbbb")

    expdir = tmp_path / "exp"
    ex = json.loads(_cli(["export", "--dir", str(expdir), "--json"], home).stdout)
    assert ex["exported"] == 2
    assert (expdir / "concepts" / "a.md").exists()
    assert (expdir / "projects" / "b.md").exists()

    home2 = _setup(tmp_path / "h2")[0]
    imp = json.loads(_cli(["import", str(expdir), "--json"], home2).stdout)
    assert imp["imported"] == 2 and imp["errors"] == []
    s1 = json.loads(_cli(["stats", "--json"], home2).stdout)
    imp2 = json.loads(_cli(["import", str(expdir), "--json"], home2).stdout)
    s2 = json.loads(_cli(["stats", "--json"], home2).stdout)
    assert s1["pages"] == 2 and s2["pages"] == 2  # delta 0 on re-import


def test_import_skips_non_allowlisted(tmp_path):
    home, root = _setup(tmp_path)
    src = tmp_path / "src"
    (src / "concepts").mkdir(parents=True)
    (src / "admin").mkdir(parents=True)
    (src / "concepts" / "ok.md").write_text("# ok\n", encoding="utf-8")
    (src / "admin" / "skip.md").write_text("# skip\n", encoding="utf-8")
    imp = json.loads(_cli(["import", str(src), "--json"], home).stdout)
    assert imp["imported"] == 1
    assert imp["skipped"] == 1
    assert imp["errors"][0]["file"] == "admin/skip.md"


def test_import_reports_bad_slug_files_and_atomic(tmp_path):
    # a filename that fails slug validation must land in `errors`, not kill
    # the batch with an uncaught SlugError (exit 2); good files still import
    home, root = _setup(tmp_path)
    src = tmp_path / "src"
    (src / "concepts").mkdir(parents=True)
    (src / "concepts" / "Bad Slug!.md").write_text("# bad\n", encoding="utf-8")
    (src / "concepts" / "good.md").write_text("# good\n", encoding="utf-8")
    imp = json.loads(_cli(["import", str(src), "--json"], home).stdout)
    assert imp["imported"] == 1 and imp["skipped"] == 1
    assert imp["errors"][0]["file"] == "concepts/Bad Slug!.md"
    g = json.loads(_cli(["get", "concepts/good", "--json"], home).stdout)
    assert "# good" in g["content"]


# ------------------------------------------------------------------ integration: backup/recover

def test_backup_recover(tmp_path):
    home, root = _setup(tmp_path)
    _cli(["put", "concepts/snap", "--no-embed", "--json"], home, input="ORIGINAL")
    bk = json.loads(_cli(["backup", "--keep", "2", "--json"], home).stdout)
    snap = Path(bk["snapshot"])
    assert snap.exists()

    # FTS triggers make keyword search live, and it must survive re-put
    # (chunk delete+insert round-trips the trigger trio).
    _cli(["put", "concepts/snap", "--no-embed", "--json"], home, input="ORIGINAL")
    d = json.loads(_cli(["doctor", "--json"], home).stdout)
    assert d["ok"] is True
    assert d["checks"]["stale_fts"] == 0
    assert d["checks"]["stale_embeds"] >= 1  # embeddings pending -> informational

    _cli(["put", "concepts/snap", "--json"], home, input="CHANGED")
    assert "CHANGED" in json.loads(_cli(["get", "concepts/snap", "--json"], home).stdout)["content"]

    # without --yes: fail closed
    deny = _cli(["recover", str(snap), "--json"], home)
    assert deny.returncode == 1

    rc = _cli(["recover", str(snap), "--yes", "--json"], home)
    assert rc.returncode == 0, rc.stderr
    assert "ORIGINAL" in json.loads(_cli(["get", "concepts/snap", "--json"], home).stdout)["content"]

    # rotation: 3 more backups, keep 2 -> at most 2 snapshots
    for _ in range(3):
        _cli(["backup", "--keep", "2", "--json"], home)
    snaps = sorted((home / ".irminsul" / "backups").glob("irminsul-*.db"))
    assert len(snaps) <= 2


# ------------------------------------------------------------------ integration: links cleanup on re-put (Phase 3 armor)

def test_reput_cleans_orphaned_link_edges(tmp_path):
    """links edges are keyed by src_chunk_id; re-chunk mints fresh ids, so old
    edges must be deleted with the chunks they referenced. No writer populates
    links yet (Phase 3) — this is armor so the writer can land without orphaning."""
    store = tmp_path / "links.db"
    conn = db.connect(store)
    assert db.load_vec(conn)
    db.migrate(conn)
    pid = pages.upsert_page(conn, "projects/net", "body about the network", tags=[])
    cid = conn.execute("SELECT id FROM chunks WHERE page_id = ?", (pid,)).fetchone()["id"]
    conn.execute("INSERT INTO links(src_chunk_id, dst_slug) VALUES (?, 'concepts/foo')", (cid,))
    conn.commit()
    assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 1
    # re-put -> old chunk deleted -> its edge must go with it
    pages.upsert_page(conn, "projects/net", "edited body entirely different", tags=[])
    assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 0
    conn.close()


def test_hard_delete_cleans_incoming_link_edges(tmp_path):
    """dst-side armor: hard-deleting a page must also drop edges whose
    dst_slug points at it (incoming links), not just its own outgoing ones."""
    store = tmp_path / "links2.db"
    conn = db.connect(store)
    assert db.load_vec(conn)
    db.migrate(conn)
    a_id = pages.upsert_page(conn, "projects/a", "links to b", tags=[])
    pages.upsert_page(conn, "concepts/b", "the target", tags=[])
    a_chunk = conn.execute("SELECT id FROM chunks WHERE page_id = ?", (a_id,)).fetchone()["id"]
    conn.execute("INSERT INTO links(src_chunk_id, dst_slug) VALUES (?, 'concepts/b')", (a_chunk,))
    conn.commit()
    assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 1
    # hard-delete the TARGET (soft delete + prune at cutoff) -> incoming edge must go
    pages.soft_delete(conn, "concepts/b")
    pages.prune(conn, 0)
    assert conn.execute("SELECT count(*) FROM links").fetchone()[0] == 0
    conn.close()


# ------------------------------------------------------------------ frontmatter surfacing (report-only)

def test_put_surfaces_dropped_frontmatter_keys(tmp_path):
    home, _ = _setup(tmp_path)
    extra = "---\ntype: note\ntitle: T\naliases: [old, x]\nstatus: draft\ncaptured_at: 2026-01-01\n---\n\nbody\n"
    out = json.loads(_cli(["put", "concepts/extra", "--json"], home, input=extra).stdout)
    assert out["dropped_keys"] == ["aliases", "captured_at", "status"]  # type/title consumed
    # clean frontmatter -> no dropped_keys key at all (additive-optional, no noise)
    clean = json.loads(_cli(["put", "concepts/clean", "--json"], home,
                            input="---\ntype: project\ntags:\n  - x\n---\n\nbody\n").stdout)
    assert "dropped_keys" not in clean


def test_import_surfaces_dropped_frontmatter_keys(tmp_path):
    home, _ = _setup(tmp_path)
    td = tmp_path / "tree"
    (td / "concepts").mkdir(parents=True)
    (td / "concepts" / "a.md").write_text(
        "---\ntags:\n  - x\naliases: [y]\n---\n\nhello\n", encoding="utf-8")
    out = json.loads(_cli(["import", str(td), "--json"], home).stdout)
    assert out["dropped"] == [{"file": "concepts/a.md", "keys": ["aliases"]}]
