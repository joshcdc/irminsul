"""Embed/rerank provider seam (Phase 2).

The tool hosts no LLM, but the retrieval pipeline needs vectors. Provider is a
config knob, not a code path: `embed.provider` dispatches to an implementation.

- `voyage`  — Voyage AI (MongoDB-owned): voyage-4-large embeddings (1024-d,
  asymmetric `input_type: query|document`) + rerank-2.5. Key env-only:
  `VOYAGE_API_KEY`. This is the live provider (chosen 2026-08-25 after
  ZeroEntropy's 2026-09-04 sunset).
- `fake`    — deterministic hash-to-unit-vector, for tests/offline. NO network.
  `embed.provider=fake` via config or `IRMINSUL_EMBED_PROVIDER=fake`.

Voyage API is OpenAI-schema-compatible for mostly-identical payloads; we call
it with httpx directly so there's exactly one network dependency.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import List, Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        """One fixed-dim vector per text. input_type query|document (asymmetric)."""
        ...

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """Relevance score per document (higher = more relevant)."""
        ...


class VoyageEmbedder:
    def __init__(self, model: str = "voyage-4-large", dim: int = 1024,
                 rerank_model: str = "rerank-2.5",
                 api_base: str = "https://api.voyageai.com"):
        self.model = model
        self.dim = dim
        self.rerank_model = rerank_model
        self.api_base = api_base.rstrip("/")
        self._key = os.environ.get("VOYAGE_API_KEY", "")
        if not self._key:
            raise RuntimeError(
                "VOYAGE_API_KEY not set — embedding needs the key from the env "
                "plane only (never config/DB). For offline tests use "
                "IRMINSUL_EMBED_PROVIDER=fake."
            )

    def embed(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        import httpx
        r = httpx.post(
            f"{self.api_base}/v1/embeddings",
            headers={"Authorization": f"Bearer {self._key}"},
            json={
                "model": self.model,
                "input": texts,
                "input_type": input_type,     # voyage-4 asymmetric query|document
                "output_dimension": self.dim,  # NEVER let the default 1024 drift from self.dim
            },
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        out = [item["embedding"] for item in data["data"]]
        # Defense in depth: cap sanity, not correctness — wrong dim corrupts vec0 later.
        if any(len(v) != self.dim for v in out):
            raise RuntimeError(f"voyage returned wrong dim: expected {self.dim}")
        return out

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        import httpx
        r = httpx.post(
            f"{self.api_base}/v1/rerank",
            headers={"Authorization": f"Bearer {self._key}"},
            json={"model": self.rerank_model, "query": query, "documents": documents},
            timeout=60.0,
        )
        r.raise_for_status()
        data = r.json()
        # Voyage rerank is OpenAI-schema-shaped: {"object":"list","data":[{index, relevance_score}]}
        results = sorted(data["data"], key=lambda d: d["index"])
        return [d["relevance_score"] for d in results]


class FakeEmbedder:
    """Deterministic unit-ish vectors (hash of text, L2-normalized) at self.dim.
    Same shape as Voyage so search/embed code paths are testable offline.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> List[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        v = [b / 255.0 - 0.5 for b in h]
        # pad/truncate to dim deterministically
        v = (v * (self.dim // len(v) + 1))[: self.dim]
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed(self, texts: List[str], input_type: str = "document") -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        q = self._vec(query)
        return [sum(a * b for a, b in zip(q, self._vec(d))) for d in documents]


def get_embedder(provider: str, model: str = "voyage-4-large", dim: int = 1024,
                 rerank_model: str = "rerank-2.5",
                 api_base: str = "https://api.voyageai.com") -> Embedder:
    """Provider seam. rerank_model is a knob (config `search.reranker.model`),
    not a code path — same story as model/dim."""
    if provider == "voyage":
        return VoyageEmbedder(model=model, dim=dim, rerank_model=rerank_model,
                              api_base=api_base)
    if provider == "fake":
        return FakeEmbedder(dim=dim)
    raise ValueError(f"embed.provider {provider!r} unknown — use voyage | fake")
