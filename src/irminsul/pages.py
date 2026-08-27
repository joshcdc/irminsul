"""Phase 1 — slug-keyed page CRUD (put/get/list/stats/delete/restore/prune).

Invariants: slugs are identity (`put` upserts, re-put = 0 row growth); two write
paths only (`put` + `import`); soft delete with a recoverable window.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from . import chunk as chunking

SLUG_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class SlugError(ValueError):
    pass


def validate_slug(slug: str) -> str:
    """Per-segment charset: `/`-separated segments, each [a-z0-9][a-z0-9._-]*."""
    slug = (slug or "").strip().strip("/")
    if not slug:
        raise SlugError("empty slug")
    for seg in slug.split("/"):
        if seg in ("", ".", ".."):
            raise SlugError(f"invalid slug segment {seg!r} in {slug!r}")
        if not SLUG_SEGMENT_RE.match(seg):
            raise SlugError(
                f"invalid segment {seg!r} in {slug!r} — each segment must match "
                "[a-z0-9][a-z0-9._-]*"
            )
    return slug


def check_namespace(slug: str, allowed) -> str:
    """Top-level prefix must be in `namespace.allow` (kills namespace sprawl)."""
    if not allowed:
        return slug
    prefix = slug.split("/", 1)[0]
    if prefix not in allowed:
        raise SlugError(
            f"namespace {prefix!r} not in allowlist — use one of: "
            f"{', '.join(sorted(allowed))}"
        )
    return slug


def _now() -> str:
    # UTC, same format as SQLite datetime('now') for string comparisons.
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_page(conn, slug, include_deleted: bool = False):
    q = "SELECT * FROM pages WHERE slug = ?"
    if not include_deleted:
        q += " AND deleted_at IS NULL"
    return conn.execute(q, (slug,)).fetchone()


def upsert_page(conn, slug, content_md, *, type="note", title=None, created=None,
                tags=None, chunk_size: int = 600, chunk_overlap: int = 60,
                commit: bool = True) -> int:
    """Write/overwrite a page by slug; re-chunk; revive if soft-deleted."""
    slug = validate_slug(slug)
    now = _now()
    row = get_page(conn, slug, include_deleted=True)
    if row is None:
        created = created or now
        cur = conn.execute(
            "INSERT INTO pages(slug, type, title, created, updated, content_md)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (slug, type or "note", title, created, now, content_md),
        )
        page_id = cur.lastrowid
    else:
        page_id = row["id"]
        keep_created = row["created"] or created or now
        conn.execute(
            "UPDATE pages SET type = ?, title = ?, created = ?, updated = ?,"
            " deleted_at = NULL, content_md = ? WHERE id = ?",
            (type or "note", title, keep_created, now, content_md, page_id),
        )
    _replace_chunks(conn, page_id, content_md, chunk_size, chunk_overlap)
    _replace_tags(conn, page_id, tags or [])
    if commit:
        conn.commit()
    return page_id


def _replace_chunks(conn, page_id, content_md, size, overlap) -> None:
    _delete_chunks(conn, page_id)
    for seq, txt in enumerate(chunking.chunk_text(content_md, size=size, overlap=overlap)):
        conn.execute(
            "INSERT INTO chunks(page_id, seq, chunk_text, embed_model) VALUES (?, ?, ?, NULL)",
            (page_id, seq, txt),
        )


def _delete_chunks(conn, page_id) -> None:
    ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE page_id = ?", (page_id,))]
    if not ids:
        return
    marks = ",".join("?" * len(ids))
    # vec0 has no UPDATE; stale recompute is delete+insert (Phase 2).
    conn.execute(f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({marks})", ids)
    # links edges are derived from content (src chunk → dst slug) — re-chunk
    # mints fresh chunk ids, so old src_chunk_id edges would orphan. Clean them
    # with the chunks they referenced (Phase 3 writer expects this to already hold).
    conn.execute(f"DELETE FROM links WHERE src_chunk_id IN ({marks})", ids)
    # chunks_fts is kept in sync by the v2 triggers (chunks_ad handles DELETE) —
    # no direct FTS writes needed here.
    conn.execute(f"DELETE FROM chunks WHERE id IN ({marks})", ids)


def _replace_tags(conn, page_id, tags) -> None:
    conn.execute("DELETE FROM tags WHERE page_id = ?", (page_id,))
    # normalize + dedupe: re-put with duplicate tags must not grow rows
    norm = (t for t in ((tag or "").strip().lower() for tag in tags) if t)
    for t in dict.fromkeys(norm):
        conn.execute("INSERT INTO tags(page_id, tag) VALUES (?, ?)", (page_id, t))


def list_pages(conn, limit: int = 1000, include_deleted: bool = False):
    q = "SELECT id, slug, type, title, created, updated, deleted_at FROM pages"
    if not include_deleted:
        q += " WHERE deleted_at IS NULL"
    q += " ORDER BY slug LIMIT ?"
    return conn.execute(q, (limit,)).fetchall()


def soft_delete(conn, slug) -> dict:
    row = get_page(conn, slug, include_deleted=True)
    if row is None:
        raise KeyError(slug)
    if row["deleted_at"]:
        return {"slug": slug, "deleted_at": row["deleted_at"]}
    now = _now()
    conn.execute("UPDATE pages SET deleted_at = ? WHERE id = ?", (now, row["id"]))
    conn.commit()
    return {"slug": slug, "deleted_at": now}


def restore(conn, slug) -> dict:
    row = get_page(conn, slug, include_deleted=True)
    if row is None:
        raise KeyError(slug)
    conn.execute("UPDATE pages SET deleted_at = NULL WHERE id = ?", (row["id"],))
    conn.commit()
    return {"slug": slug, "deleted_at": None}


def prune(conn, older_than_hours: float = 72) -> int:
    """Permanently purge soft-deleted pages (+ their chunks/embeddings/tags/links).

    Uses `<=` (inclusive cutoff): a page deleted in the same second as the
    cutoff IS beyond the window (matching-now rows must still be collectible).
    """
    rows = conn.execute(
        "SELECT id FROM pages WHERE deleted_at IS NOT NULL"
        " AND deleted_at <= datetime('now', ?)",
        (f"-{float(older_than_hours):g} hours",),
    ).fetchall()
    for r in rows:
        _hard_delete(conn, r["id"])
    conn.commit()
    return len(rows)


def _hard_delete(conn, page_id) -> None:
    ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE page_id = ?", (page_id,))]
    if ids:
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({marks})", ids)
        # FTS stays in sync via the chunks_ad trigger.
        conn.execute(f"DELETE FROM chunks WHERE id IN ({marks})", ids)
    conn.execute("DELETE FROM tags WHERE page_id = ?", (page_id,))
    # outgoing edges (chunks of this page) — re-put armor covers the same on _delete_chunks
    conn.execute("DELETE FROM links WHERE src_chunk_id IN"
                 " (SELECT id FROM chunks WHERE page_id = ?)", (page_id,))
    # incoming edges (other chunks linking TO this page's slug) — a dead target
    # must not leave dangling dst-side rows. Slug read before the page row dies.
    slug = conn.execute("SELECT slug FROM pages WHERE id = ?", (page_id,)).fetchone()
    if slug:
        conn.execute("DELETE FROM links WHERE dst_slug = ?", (slug["slug"],))
    conn.execute("DELETE FROM pages WHERE id = ?", (page_id,))


def tags_for(conn, page_id) -> list:
    return [r["tag"] for r in conn.execute(
        "SELECT tag FROM tags WHERE page_id = ? ORDER BY tag", (page_id,))]


def stats(conn) -> dict:
    pages = conn.execute("SELECT count(*) FROM pages").fetchone()[0]
    active = conn.execute("SELECT count(*) FROM pages WHERE deleted_at IS NULL").fetchone()[0]
    chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    tags = conn.execute("SELECT count(*) FROM tags").fetchone()[0]
    return {"pages": pages, "active": active, "deleted": pages - active,
            "chunks": chunks, "tags": tags}
