"""Semantic (dense vector) retrieval.

What it does
------------
Embeds the query with the same embedder used at ingest time, then asks the vector
store for the top-k most cosine-similar chunks. Populates each candidate's
``similarity`` and ``vector_distance`` for the trace.

Why it exists
-------------
Dense retrieval captures meaning/paraphrase that keyword search misses ("factory
shutdown" vs "restricted period"). It is the default retriever.

Limitations
-----------
- Purely semantic: an exact article number or rare token can be under-weighted.
  Combine with BM25 via the hybrid retriever when that matters.
"""

from __future__ import annotations

from ..errors import ConfigurationError
from ..models import RetrievedCandidate
from .base import RetrievalDeps, Retriever, retriever_registry


@retriever_registry.register("vector")
class VectorRetriever(Retriever):
    name = "vector"

    def __init__(self, deps: RetrievalDeps) -> None:
        if deps.store is None or deps.embedder is None:
            raise ConfigurationError("vector retriever requires a store and an embedder")
        self._store = deps.store
        self._embedder = deps.embedder

    def retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedCandidate]:
        q_vec = self._embedder.embed_query(query)
        hits = self._store.search(q_vec, k=k, metadata_filter=metadata_filter)
        candidates: list[RetrievedCandidate] = []
        for rank, hit in enumerate(hits):
            candidates.append(
                RetrievedCandidate(
                    chunk=hit.chunk,
                    score=hit.similarity,  # primary ranking score = cosine similarity
                    similarity=hit.similarity,
                    vector_distance=hit.distance,
                    rank=rank,
                    retriever=self.name,
                )
            )
        return candidates
