"""Embedder interface + normalization helper.

Vector normalization (why it matters)
--------------------------------------
Cosine similarity between vectors ``a`` and ``b`` is::

    cos(a, b) = (a . b) / (||a|| * ||b||)

If we L2-normalize every vector up front (divide by its Euclidean norm so
``||v|| == 1``), then cosine similarity reduces to a plain dot product::

    cos(a, b) = a . b            # when ||a|| == ||b|| == 1

That is a big deal for retrieval: with normalized vectors, a whole matrix of
similarities is just one matrix multiplication ``matrix @ query`` (see the NumPy
store). We normalize here, once, so every downstream index can rely on it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..registry import Registry

embedder_registry: Registry["Embedder"] = Registry("embedder")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Return a copy of ``matrix`` with each row scaled to unit L2 norm.

    ``matrix`` shape is (n_vectors, dim). We compute each row's norm and divide,
    guarding against divide-by-zero for empty vectors.
    """

    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)  # shape (n, 1)
    norms = np.where(norms == 0.0, 1.0, norms)  # avoid 0/0 -> nan
    return matrix / norms


@runtime_checkable
class Embedder(Protocol):
    """Encodes text into fixed-dimension, L2-normalized vectors."""

    name: str
    dim: int

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Return an (len(texts), dim) float32 array of normalized embeddings."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Return a single (dim,) normalized embedding for a query."""
        ...
