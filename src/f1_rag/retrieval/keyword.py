"""Keyword retrieval with BM25 (Okapi).

What it does
------------
Builds a BM25 index over chunk tokens and scores chunks by lexical overlap with the
query. Applies metadata filtering explicitly after scoring.

Why it exists
-------------
BM25 is a strong sparse baseline and complements dense retrieval: it nails exact
terms, article numbers, and rare vocabulary that embeddings may blur. It is also a
useful standalone comparison point.

How BM25 works (brief)
----------------------
BM25 scores a document for a query by summing, over query terms, an IDF weight
(rare terms count more) times a term-frequency saturation function (repeats help,
with diminishing returns) normalized by document length. ``rank_bm25`` implements
this; we just tokenize consistently.

Limitations
-----------
- No semantics: paraphrases score poorly. Use the hybrid retriever to get both.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ..models import Chunk, RetrievedCandidate
from .base import RetrievalDeps, Retriever, retriever_registry
from .base import min_max_normalize  # noqa: F401  (re-exported for convenience)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization shared by index + query."""

    return _TOKEN_RE.findall(text.lower())


@retriever_registry.register("bm25")
class KeywordRetriever(Retriever):
    name = "bm25"

    def __init__(self, deps: RetrievalDeps) -> None:
        self._chunks: list[Chunk] = list(deps.chunks)
        self._corpus_tokens = [tokenize(c.text) for c in self._chunks]
        # Guard against an all-empty corpus (BM25Okapi requires non-empty docs).
        safe = [toks or ["\u0000"] for toks in self._corpus_tokens]
        self._bm25 = BM25Okapi(safe) if self._chunks else None

    def retrieve(
        self,
        query: str,
        k: int,
        metadata_filter: dict | None = None,
    ) -> list[RetrievedCandidate]:
        if self._bm25 is None:
            return []
        from ..indexing.base import matches_filter

        scores = self._bm25.get_scores(tokenize(query))
        indexed = [
            (i, float(s))
            for i, s in enumerate(scores)
            if matches_filter(self._chunks[i], metadata_filter)
        ]
        indexed.sort(key=lambda t: t[1], reverse=True)
        top = indexed[:k]
        candidates: list[RetrievedCandidate] = []
        for rank, (i, score) in enumerate(top):
            candidates.append(
                RetrievedCandidate(
                    chunk=self._chunks[i],
                    score=score,  # primary ranking score = BM25
                    keyword_score=score,
                    rank=rank,
                    retriever=self.name,
                )
            )
        return candidates
