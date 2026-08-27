"""Phase 1 — frontmatter + import (idempotent slug-upsert) / export (DB -> md)."""

from __future__ import annotations

import os
from pathlib import Path

from . import pages

# Keys upsert_page actually consumes from frontmatter. Anything else the parser
# collects is REPORT-ONLY (surfaced in put/import output, never stored) — the
# schema has no drawers for foreign metadata and we don't pretend otherwise.
FRONTMATTER_KEYS = {"type", "title", "created", "tags"}


def parse_frontmatter(text: str):
    """Return (meta dict, body). Frontmatter must start/end on their own `---` line."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = None
    for i, ln in enumerate(lines[1:], start=1):
        if ln.strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    meta = _parse_block(lines[1:end])
    body = "\n".join(lines[end + 1:]).lstrip("\n")
    return meta, body


def _parse_block(lines) -> dict:
    meta = {}
    i = 0
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        if not stripped or stripped.startswith("#") or ":" not in ln:
            i += 1
            continue
        key, _, val = ln.partition(":")
        key, val = key.strip(), val.strip()
        if not val:
            items = []
            j = i + 1
            while j < len(lines) and lines[j].strip().startswith("-"):
                items.append(lines[j].strip()[1:].strip().strip('"').strip("'"))
                j += 1
            if items:
                meta[key] = items
                i = j
                continue
        meta[key] = val.strip('"').strip("'")
        i += 1
    return meta


def build_frontmatter(type="note", title=None, created=None, updated=None, tags=()) -> str:
    lines = ["---", f"type: {type or 'note'}"]
    if title:
        lines.append(f"title: {title}")
    lines.append(f"created: {created}")
    lines.append(f"updated: {updated}")
    tags = list(tags)
    if tags:
        lines.append("tags:")
        lines.extend(f"  - {t}" for t in tags)
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def slug_from_rel(rel: Path) -> str:
    return str(rel).replace("\\", "/")[:-3] if str(rel).endswith(".md") else str(rel).replace("\\", "/")


def import_dir(conn, root, allowed=None, dry_run: bool = False) -> dict:
    """Batch slug-upsert ingest. IDEMPOTENT — re-runs overwrite in place.

    Files with frontmatter provide type/title/created/tags. Files outside the
    namespace allowlist are skipped (fail closed) and reported in `errors`.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(p for p in root.rglob("*.md"))
    imported = 0
    errors = []
    dropped = []
    for p in files:
        rel = p.relative_to(root)
        try:
            slug = pages.validate_slug(slug_from_rel(rel))
            pages.check_namespace(slug, allowed)
        except pages.SlugError as e:
            errors.append({"file": rel.as_posix(), "error": str(e)})
            continue
        if dry_run:
            imported += 1
            continue
        text = p.read_text(encoding="utf-8")
        meta, body = parse_frontmatter(text)
        dkeys = sorted(set(meta) - FRONTMATTER_KEYS)
        if dkeys:
            dropped.append({"file": rel.as_posix(), "keys": dkeys})
        pages.upsert_page(conn, slug, body, commit=False,
                          type=meta.get("type", "note"),
                          title=meta.get("title"),
                          created=meta.get("created"),
                          tags=meta.get("tags") or [])
        imported += 1
    conn.commit()  # one atomic transaction for the whole batch
    res = {"imported": imported, "skipped": len(errors), "errors": errors}
    if dropped:
        res["dropped"] = dropped  # report-only: keys parsed but not consumed
    return res


def export_dir(conn, out_dir: str) -> dict:
    """DB -> markdown tree derived from slug prefixes (the human view)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    rows = conn.execute(
        "SELECT id, slug, type, title, created, updated, content_md FROM pages"
        " WHERE deleted_at IS NULL ORDER BY slug"
    ).fetchall()
    for row in rows:
        dest = out / Path(str(row["slug"]).replace("/", os.sep) + ".md")
        dest.parent.mkdir(parents=True, exist_ok=True)
        fm = build_frontmatter(
            type=row["type"], title=row["title"],
            created=row["created"], updated=row["updated"],
            tags=pages.tags_for(conn, row["id"]),
        )
        dest.write_text(fm + (row["content_md"] or ""), encoding="utf-8")
        n += 1
    return {"exported": n, "dir": str(out)}
