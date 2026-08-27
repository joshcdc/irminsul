"""Phase 2 — irminsul search (hybrid FTS5 + vec0 KNN, RRF fusion, rerank hook).

Query-time pipeline: tokenize the query for FTS5, embed it for vec0 KNN,
fuse the two rank lists with Reciprocal Rank Fusion, then optionally rerank
the top candidates with the configured reranker (search.reranker.model).
Soft-deleted pages are excluded everywhere. Read-only — nothing here writes.
Keyword search always works (vector leg silently empty when embeddings are
missing); the vector leg requires an embedder.
"""

from __future__ import annotations

import json
import re


def build_fts_query(text: str) -> str:
    """Tokenize free text into a safe FTS5 OR-of-quoted-terms query.

    Quoting each term prevents FTS5 operators/characters injecting into the
    MATCH expression; punctuation/symbols are dropped. Empty string when no
    usable terms (callers treat that as 'no keyword leg').
    """
    words = re.findall(r"[A-Za-z0-9_]+", text or "")
    return " OR ".join(f'"{w}"' for w in words)


def hybrid_search(conn, embedder, query, limit: int = 5, rrf_k: int = 60,
                  w_fs: float = 1.0, w_vec: float = 1.0, rerank: bool = True):
    """Hybrid retrieval -> list of result dicts (highest score first):

        {chunk_id, slug, title, type, score, excerpt}

    - FTS leg: bm25-ranked, requires at least one query term.
    - vec leg: vec0 KNN by query embedding (skipped when embedder is None).
    - RRF fuses the two ordering with weights w_fs / w_vec at constant rrf_k.
    - rerank (enabled + embedder present) reorders the top candidates by the
      provider's reranker; on failure falls back to RRF order.
    - Soft-deleted pages never appear (JOIN pages, deleted_at IS NULL).
    """
    candidate_n = max(limit * 10, limit)  # widen before RRF + rerank

    # --- FTS5 keyword leg -------------------------------------------------
    fts_hits = []
    ftq = build_fts_query(query)
    if ftq:
        fts_hits = [
            (r["id"], r)
            for r in conn.execute(
                "SELECT c.id, c.chunk_text, p.slug, p.title, p.type FROM chunks_fts"
                " JOIN chunks c ON c.id = chunks_fts.rowid"
                " JOIN pages p ON p.id = c.page_id"
                " WHERE chunks_fts MATCH ? AND p.deleted_at IS NULL"
                " ORDER BY bm25(chunks_fts) LIMIT ?", (ftq, candidate_n))
        ]

    # --- vector leg (vec0 KNN) --------------------------------------------
    vec_hits = []
    if embedder is not None:
        try:
            qvec = embedder.embed([query], input_type="query")[0]
            vec_hits = [
                (r["id"], r)
                for r in conn.execute(
                    "SELECT knn.chunk_id AS id, c.chunk_text, p.slug, p.title, p.type"
                    " FROM (SELECT chunk_id FROM chunk_embeddings"
                    "       WHERE embedding MATCH ? AND k = ?) AS knn"
                    " JOIN chunks c ON c.id = knn.chunk_id"
                    " JOIN pages p ON p.id = c.page_id"
                    " WHERE p.deleted_at IS NULL", (json.dumps(qvec), candidate_n))
            ]
        except Exception:
            vec_hits = []  # no embeddings / provider hiccup -> keyword-only

    if not fts_hits and not vec_hits:
        return []

    info = {cid: n for cid, n in fts_hits}
    info.update({cid: n for cid, n in vec_hits})

    # --- RRF fusion --------------------------------------------------------
    fused = {}
    for lst, w in (([cid for cid, _ in fts_hits], w_fs),
                   ([cid for cid, _ in vec_hits], w_vec)):
        for rank, cid in enumerate(lst):
            fused[cid] = fused.get(cid, 0.0) + w / (rrf_k + rank + 1)

    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    # --- rerank top candidates (optional) ----------------------------------
    if rerank and embedder is not None and ordered:
        top = ordered[: max(limit * 2, limit)]
        try:
            cand_texts = [info[cid]["chunk_text"] for cid, _ in top]
            scores = embedder.rerank(query, cand_texts)
            ranked = sorted(zip([cid for cid, _ in top], scores),
                            key=lambda p: p[1], reverse=True)
            return [
                {"chunk_id": cid, "slug": info[cid]["slug"], "title": info[cid]["title"],
                 "type": info[cid]["type"], "score": round(float(s), 6),
                 "excerpt": info[cid]["chunk_text"]}
                for cid, s in ranked[:limit]
            ]
        except Exception:
            pass  # rerank failure -> keep pure RRF order

    return [
        {"chunk_id": cid, "slug": info[cid]["slug"], "title": info[cid]["title"],
         "type": info[cid]["type"], "score": round(fused[cid], 6),
         "excerpt": info[cid]["chunk_text"]}
        for cid, _ in ordered[:limit]
    ]
