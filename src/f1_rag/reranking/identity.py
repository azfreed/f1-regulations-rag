"""Identity reranker (no-op, the default ``none``).

What it does
------------
Returns candidates in their existing order, copying each candidate's primary score
into ``rerank_score`` so downstream code and the trace have a consistent field.

Why it exists
-------------
It makes "no reranking" a first-class, explicit choice rather than a special case,
and gives the pipeline a stable interface to fill in later with a real
cross-encoder reranker.
"""

from __future__ import annotations

from ..models import RetrievedCandidate
from .base import Reranker, reranker_registry


@reranker_registry.register("none")
class IdentityReranker(Reranker):
    name = "none"

    def rerank(
        self, query: str, candidates: list[RetrievedCandidate]
    ) -> list[RetrievedCandidate]:
        for cand in candidates:
            cand.rerank_score = cand.score
        return candidates
