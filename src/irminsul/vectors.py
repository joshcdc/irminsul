"""vec0 vector writes (Phase 2). vec0 has NO UPDATE — every recompute is
delete+insert. Keeps `chunks.embed_model` as the staleness marker (`embed --stale`
re-embeds rows where it's NULL or != the configured model)."""

from __future__ import annotations

import json


def add_embeddings(conn, chunk_ids, vectors, embed_model: str) -> int:
    """Insert embeddings for chunk ids (delete+insert semantics).

    Sets chunks.embed_model on the chunk row so `embed --stale` can find rows
    that missed embedding (crashed batch) or were written under an old model
    (provider/model bump -> everything under the old name is stale).
    """
    assert len(chunk_ids) == len(vectors), "chunk_ids/vectors length mismatch"
    if not chunk_ids:
        return 0
    marks = ",".join("?" * len(chunk_ids))
    # vec0 can't UPDATE in place: delete existing rows, insert fresh ones.
    conn.execute(f"DELETE FROM chunk_embeddings WHERE chunk_id IN ({marks})", chunk_ids)
    rows = [(cid, json.dumps(v)) for cid, v in zip(chunk_ids, vectors)]
    conn.executemany(
        "INSERT INTO chunk_embeddings(chunk_id, embedding) VALUES (?, ?)", rows
    )
    conn.execute(
        f"UPDATE chunks SET embed_model = ? WHERE id IN ({marks})",
        (embed_model, *chunk_ids),
    )
    conn.commit()
    return len(rows)


def stale_chunk_ids(conn, embed_model: str) -> list:
    """Chunk ids missing embeddings or embedded under a different model."""
    return [r["id"] for r in conn.execute(
        "SELECT c.id FROM chunks c LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id"
        " WHERE e.chunk_id IS NULL OR c.embed_model IS NULL OR c.embed_model != ?",
        (embed_model,),
    )]
