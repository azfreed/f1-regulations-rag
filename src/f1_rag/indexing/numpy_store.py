"""NumPy vector store - the transparent reference implementation.

What it does
------------
Keeps the embedding matrix in a single NumPy array and computes cosine similarity
with one matrix multiplication. Persists to ``.npz`` (vectors) + ``.jsonl`` (chunk
metadata).

Why it exists
-------------
So the vector-search mechanics stay fully visible. Chroma (and every other vector
DB) does essentially this under the hood; here you can read every line.

The math, step by step
-----------------------
Let ``M`` be the (n_chunks, dim) matrix of *L2-normalized* chunk embeddings and
``q`` the (dim,) *L2-normalized* query embedding. Because the rows of ``M`` and the
query are unit vectors:

    cosine_similarity(row_i, q) = row_i . q          # dot product
    all similarities            = M @ q              # one matrix-vector multiply

``M @ q`` yields an (n_chunks,) vector of similarities. We take the top-k largest
(``np.argpartition`` for speed, then sort those k), which is the "ranking" step.
Cosine distance is just ``1 - similarity``.

Limitations
-----------
- Brute-force O(n_chunks * dim) per query. Fine for these regulations (tens of
  thousands of chunks at most); a real deployment would use an ANN index.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..errors import IndexError_
from ..logging_utils import get_logger
from ..models import Chunk
from .base import MetadataFilter, StoreHit, VectorStore, index_registry, matches_filter

logger = get_logger(__name__)


@index_registry.register("numpy")
class NumpyVectorStore(VectorStore):
    name = "numpy"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray = np.zeros((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise IndexError_(
                f"chunk/embedding count mismatch: {len(chunks)} vs {embeddings.shape[0]}"
            )
        self._chunks = list(chunks)
        # Store as float32 and assume rows are already L2-normalized by the embedder.
        self._matrix = np.asarray(embeddings, dtype=np.float32)
        logger.info(
            "numpy store built: %d vectors, dim=%d",
            self._matrix.shape[0],
            self._matrix.shape[1] if self._matrix.ndim == 2 else 0,
        )

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[StoreHit]:
        if self._matrix.size == 0:
            return []

        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)

        # --- the whole search in three lines ---
        # 1) similarities for every chunk at once (matrix multiplication)
        sims = self._matrix @ q  # shape (n_chunks,)

        # 2) filtering: mask out chunks that fail the metadata filter by forcing
        #    their similarity to -inf so they can never enter the top-k.
        if metadata_filter:
            mask = np.array(
                [matches_filter(c, metadata_filter) for c in self._chunks], dtype=bool
            )
            sims = np.where(mask, sims, -np.inf)
            n_eligible = int(mask.sum())
        else:
            n_eligible = len(self._chunks)

        if n_eligible == 0:
            return []

        # 3) ranking: grab the top-k indices, then sort just those by similarity.
        k = min(k, n_eligible)
        top_idx = np.argpartition(-sims, k - 1)[:k]  # unsorted top-k (fast)
        top_idx = top_idx[np.argsort(-sims[top_idx])]  # sort the k by similarity

        hits: list[StoreHit] = []
        for i in top_idx:
            sim = float(sims[i])
            if sim == -np.inf:
                continue
            hits.append(
                StoreHit(
                    chunk=self._chunks[int(i)],
                    similarity=sim,
                    distance=1.0 - sim,  # cosine distance
                    index=int(i),
                )
            )
        return hits

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path / "vectors.npz", matrix=self._matrix)
        with (path / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self._chunks:
                fh.write(c.model_dump_json())
                fh.write("\n")
        logger.info("numpy store saved to %s", path)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        data = np.load(path / "vectors.npz")
        self._matrix = data["matrix"].astype(np.float32)
        self._chunks = []
        with (path / "chunks.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self._chunks.append(Chunk.model_validate(json.loads(line)))
        logger.info("numpy store loaded from %s (%d vectors)", path, len(self._chunks))
