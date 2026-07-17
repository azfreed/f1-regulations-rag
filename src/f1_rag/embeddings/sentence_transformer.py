"""Sentence-Transformers embedder (default: all-MiniLM-L6-v2).

What it does
------------
Wraps a Sentence-Transformers model to produce semantic embeddings, then
L2-normalizes them so downstream cosine similarity is a dot product.

Why it exists
-------------
MiniLM is small, fast, and a strong general-purpose retrieval model - a sensible
default for a local test bed. Registered as ``minilm``.

Assumptions
-----------
- The optional ``sentence-transformers`` dependency is installed. If not, importing
  this module fails and the package falls back to the hashing embedder.
- First use downloads the model to the HuggingFace cache (needs network once).

Replaceable alternatives
-------------------------
- Any other model id can be passed to the constructor; swap the whole class out for
  an API-based embedder without touching downstream stages.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from ..logging_utils import get_logger
from .base import Embedder, embedder_registry, l2_normalize

logger = get_logger(__name__)


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        logger.info("loading sentence-transformer model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        self.name = "minilm"
        # The accessor was renamed across sentence-transformers versions.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # Batch encoding keeps memory bounded and is much faster than per-text calls.
        raw = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return l2_normalize(raw)

    def embed_query(self, text: str) -> np.ndarray:
        raw = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)
        return l2_normalize(raw)[0]


@embedder_registry.register("minilm")
def _make_minilm(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    return SentenceTransformerEmbedder(model_name)
