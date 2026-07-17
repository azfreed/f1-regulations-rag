from __future__ import annotations

import numpy as np

from f1_rag.embeddings.base import l2_normalize
from f1_rag.embeddings.hashing import HashingEmbedder
from f1_rag.indexing.numpy_store import NumpyVectorStore
from f1_rag.models import RegulationSection
from tests.conftest import make_chunk


def test_l2_normalize_unit_norm():
    m = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    out = l2_normalize(m)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.isclose(np.linalg.norm(out[1]), 0.0)  # zero vector stays zero, no nan
    assert not np.isnan(out).any()


def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=64)
    v1 = emb.embed(["pit lane speed limit"])
    v2 = emb.embed(["pit lane speed limit"])
    assert np.allclose(v1, v2)
    assert np.isclose(np.linalg.norm(v1[0]), 1.0)


def test_numpy_store_cosine_matches_manual():
    emb = HashingEmbedder(dim=64)
    chunks = [
        make_chunk("c1", "pit lane speed limit during the race"),
        make_chunk("c2", "cost cap financial regulations for teams", section=RegulationSection.D),
        make_chunk("c3", "fillet radius technical bodywork definition", section=RegulationSection.C),
    ]
    vectors = emb.embed([c.text for c in chunks])
    store = NumpyVectorStore()
    store.build(chunks, vectors)

    q = emb.embed_query("pit lane speed")
    hits = store.search(q, k=3)
    # The most similar chunk should be c1; verify similarity equals a manual dot product.
    assert hits[0].chunk.chunk_id == "c1"
    manual = float(vectors[0] @ q)
    assert np.isclose(hits[0].similarity, manual, atol=1e-5)
    # distance = 1 - similarity
    assert np.isclose(hits[0].distance, 1.0 - hits[0].similarity, atol=1e-6)


def test_numpy_store_metadata_filter():
    emb = HashingEmbedder(dim=64)
    chunks = [
        make_chunk("a", "general provisions", section=RegulationSection.A),
        make_chunk("d", "cost cap", section=RegulationSection.D),
    ]
    store = NumpyVectorStore()
    store.build(chunks, emb.embed([c.text for c in chunks]))
    hits = store.search(emb.embed_query("cost cap"), k=5, metadata_filter={"section": "D"})
    assert {h.chunk.chunk_id for h in hits} == {"d"}


def test_numpy_store_save_load(tmp_path):
    emb = HashingEmbedder(dim=32)
    chunks = [make_chunk("x", "some regulation text")]
    store = NumpyVectorStore()
    store.build(chunks, emb.embed([c.text for c in chunks]))
    store.save(tmp_path / "idx")

    loaded = NumpyVectorStore()
    loaded.load(tmp_path / "idx")
    assert len(loaded) == 1
    assert loaded.search(emb.embed_query("regulation"), k=1)[0].chunk.chunk_id == "x"
