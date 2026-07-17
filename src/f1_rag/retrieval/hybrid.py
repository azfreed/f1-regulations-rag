"""Hybrid retrieval: combine dense (vector) and sparse (BM25) scores.

What it does
------------
Runs both the vector and BM25 retrievers over a widened candidate pool, normalizes
each score type to [0, 1], and combines them with a weighted sum. Deduplicates by
``chunk_id`` (a chunk found by both retrievers appears once, with both raw scores
preserved) and returns the top-k by combined score.

Why it exists
-------------
Dense and sparse retrieval fail in different ways; combining them is usually more
robust than either alone. This retriever is the recommended default for real use.

Score combination (the important part)
---------------------------------------
Raw cosine similarities (~0..1) and BM25 scores (unbounded, corpus-dependent) are
not comparable, so we min-max normalize each set independently, then::

    combined = alpha * norm_similarity + (1 - alpha) * norm_keyword

``alpha`` defaults to 0.5. Deduplication keeps the max-normalized value per source
so a chunk isn't penalized for being missing from one retriever's shortlist.

Limitations
-----------
- Min-max normalization is query-local (depends on the current candidate pool); it
  is simple and transparent but not calibrated across queries.
"""

from __future__ import annotations

from ..models import RetrievedCandidate
from .base import RetrievalDeps, Retriever, min_max_normalize, retriever_registry
from .keyword import KeywordRetriever
from .vector import VectorRetriever


@retriever_registry.register("hybrid")
class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, deps: RetrievalDeps, alpha: float = 0.5, pool_multiplier: int = 4) -> None:
        self._vector = VectorRetriever(deps)
        self._keyword = KeywordRetriever(deps)
        self._alpha = alpha
        self._pool_multiplier = pool_multiplier

    def retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedCandidate]:
        # Widen each retriever's pool so the fusion has candidates to work with.
        pool = max(k * self._pool_multiplier, k)
        vec = self._vector.retrieve(query, pool, metadata_filter)
        kw = self._keyword.retrieve(query, pool, metadata_filter)

        # Normalize each score type independently onto [0, 1].
        vec_norm = min_max_normalize([c.similarity or 0.0 for c in vec])
        kw_norm = min_max_normalize([c.keyword_score or 0.0 for c in kw])

        # Merge by chunk_id, preserving raw scores from whichever retriever saw it.
        merged: dict[str, RetrievedCandidate] = {}
        norm_sim: dict[str, float] = {}
        norm_kw: dict[str, float] = {}

        for cand, n in zip(vec, vec_norm):
            cid = cand.chunk.chunk_id
            merged[cid] = RetrievedCandidate(
                chunk=cand.chunk,
                score=0.0,
                similarity=cand.similarity,
                vector_distance=cand.vector_distance,
                retriever=self.name,
            )
            norm_sim[cid] = n

        for cand, n in zip(kw, kw_norm):
            cid = cand.chunk.chunk_id
            if cid not in merged:
                merged[cid] = RetrievedCandidate(
                    chunk=cand.chunk, score=0.0, retriever=self.name
                )
            merged[cid].keyword_score = cand.keyword_score
            norm_kw[cid] = n

        # Weighted combination (missing component contributes 0 after normalization).
        for cid, cand in merged.items():
            cand.score = self._alpha * norm_sim.get(cid, 0.0) + (1 - self._alpha) * norm_kw.get(cid, 0.0)

        ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)[:k]
        for rank, cand in enumerate(ranked):
            cand.rank = rank
        return ranked
