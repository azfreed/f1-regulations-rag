"""Deterministic hashing embedder (offline, dependency-free fallback).

What it does
------------
Encodes text into a fixed-dim vector using the hashing trick: each token is hashed
to a bucket and (optionally signed) counts accumulate into that bucket, then the
vector is L2-normalized. No model download, no network, fully deterministic.

Why it exists
-------------
So the whole pipeline - chunking, indexing, retrieval, tracing, evaluation - runs
end-to-end with zero heavy dependencies and reproducible output (great for unit
tests and for demonstrating the vector-search mechanics). It is a bag-of-words
model, so semantic quality is limited; use ``minilm`` for real retrieval.

Limitations
-----------
- No semantics: synonyms and paraphrases are not close in this space.

Replaceable alternatives
-------------------------
- :mod:`f1_rag.embeddings.sentence_transformer` (``minilm``) for semantic vectors.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from ..logging_utils import get_logger
from .base import Embedder, embedder_registry, l2_normalize

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@embedder_registry.register("hashing")
class HashingEmbedder(Embedder):
    name = "hashing"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN_RE.findall(text.lower()):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
            bucket = h % self.dim
            sign = 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collisions
            vec[bucket] += sign
        return vec

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self._vector(t) for t in texts])
        return l2_normalize(matrix)

    def embed_query(self, text: str) -> np.ndarray:
        return l2_normalize(self._vector(text).reshape(1, -1))[0]
