"""Chroma vector store (alternative to the NumPy store).

What it does
------------
Persists chunk vectors + metadata in a local Chroma collection and delegates
nearest-neighbour search to Chroma. Produces the same :class:`StoreHit` output as
the NumPy store, so retrieval code is identical regardless of backend.

Why it exists
-------------
To demonstrate that the index is a swappable component and to offer a "real" vector
DB path. Chroma uses cosine space here (configured on the collection).

Assumptions
-----------
- The optional ``chromadb`` dependency is installed. Vectors are pre-normalized by
  the embedder; we still set the collection distance to cosine for correctness.

Limitations
-----------
- Chroma's internals are opaque compared to the NumPy store; use the NumPy store
  when you want to see the mechanics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import chromadb

from ..logging_utils import get_logger
from ..models import Chunk, RegulationSection
from .base import MetadataFilter, StoreHit, VectorStore, index_registry

logger = get_logger(__name__)

_COLLECTION = "f1_regulations"


def _chunk_metadata(chunk: Chunk) -> dict:
    # Chroma metadata values must be scalar; flatten the fields we filter/cite on.
    return {
        "section": chunk.section.value,
        "article_number": chunk.article_number,
        "parent_article": chunk.parent_article or "",
        "article_heading": chunk.article_heading or "",
        "pdf_page_number": chunk.pdf_page_number,
        "page_label": chunk.page_label or "",
        "source_filename": chunk.source_filename,
        "document_title": chunk.document_title,
        "issue_number": chunk.issue_number or -1,
        "publication_date": chunk.publication_date or "",
        "page_image_path": chunk.page_image_path or "",
        "chunk_index": chunk.chunk_index,
    }


def _chunk_from_doc(doc: str, md: dict) -> Chunk:
    return Chunk(
        chunk_id=md.get("chunk_id", ""),
        text=doc,
        section=RegulationSection(md["section"]),
        article_number=md["article_number"],
        parent_article=md.get("parent_article") or None,
        article_heading=md.get("article_heading") or None,
        pdf_page_number=int(md["pdf_page_number"]),
        page_label=md.get("page_label") or None,
        source_filename=md["source_filename"],
        document_title=md["document_title"],
        issue_number=(int(md["issue_number"]) if md.get("issue_number", -1) != -1 else None),
        publication_date=md.get("publication_date") or None,
        page_image_path=md.get("page_image_path") or None,
        chunk_index=int(md.get("chunk_index", 0)),
    )


@index_registry.register("chroma")
class ChromaVectorStore(VectorStore):
    name = "chroma"

    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._persist_path = str(persist_path) if persist_path else None
        self._client = None
        self._collection = None

    def __len__(self) -> int:
        return self._collection.count() if self._collection is not None else 0

    def _ensure_client(self, path: str) -> None:
        self._client = chromadb.PersistentClient(path=path)

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        path = self._persist_path or ".chroma"
        Path(path).mkdir(parents=True, exist_ok=True)
        self._ensure_client(path)
        # Recreate the collection for idempotent rebuilds.
        try:
            self._client.delete_collection(_COLLECTION)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass
        self._collection = self._client.create_collection(
            _COLLECTION, metadata={"hnsw:space": "cosine"}
        )
        metadatas = []
        for c in chunks:
            md = _chunk_metadata(c)
            md["chunk_id"] = c.chunk_id
            metadatas.append(md)
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[e.tolist() for e in np.asarray(embeddings, dtype=np.float32)],
            metadatas=metadatas,
        )
        logger.info("chroma store built: %d vectors at %s", len(chunks), path)

    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[StoreHit]:
        if self._collection is None:
            return []
        where = None
        if metadata_filter:
            # Translate our filter into Chroma's where clause (equality only here).
            where = {key: (val.value if hasattr(val, "value") else val) for key, val in metadata_filter.items()}
        res = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype=np.float32).tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[StoreHit] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, (doc, md, dist) in enumerate(zip(docs, metas, dists)):
            similarity = 1.0 - float(dist)  # chroma cosine distance -> similarity
            hits.append(
                StoreHit(chunk=_chunk_from_doc(doc, md), similarity=similarity, distance=float(dist), index=i)
            )
        return hits

    def save(self, path: str | Path) -> None:
        # Chroma persists on write when using PersistentClient; nothing extra needed.
        logger.info("chroma store persists automatically at %s", self._persist_path)

    def load(self, path: str | Path) -> None:
        persist = self._persist_path or str(path)
        self._ensure_client(persist)
        self._collection = self._client.get_collection(_COLLECTION)
        logger.info("chroma store loaded from %s (%d vectors)", persist, len(self))
