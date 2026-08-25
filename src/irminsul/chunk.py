"""Phase 1 — text chunking (size/overlap), re-chunk on put."""


def chunk_text(text: str, size: int = 600, overlap: int = 60) -> list:
    """Split text into ~size-char chunks with ~overlap of context at boundaries.

    Prefers splitting at a newline/space boundary near the tail of each chunk
    (up to ~25% backoff) so chunks stay readable. Deterministic for a given input.
    """
    if size <= 0:
        raise ValueError("chunk.size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("chunk.overlap must be >= 0 and < chunk.size")
    text = text or ""
    n = len(text)
    if n <= size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    backoff = max(size // 4, 1)
    while start < n:
        end = min(start + size, n)
        if end < n:
            nl = text.rfind("\n", start + backoff, end)
            sp = text.rfind(" ", start + backoff, end)
            cut = max(nl, sp)
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
