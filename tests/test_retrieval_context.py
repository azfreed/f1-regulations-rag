from __future__ import annotations

from f1_rag.context.assembly import GreedyContextAssembler
from f1_rag.embeddings.hashing import HashingEmbedder
from f1_rag.indexing.numpy_store import NumpyVectorStore
from f1_rag.models import RegulationSection, RetrievedCandidate
from f1_rag.retrieval.base import RetrievalDeps, min_max_normalize, normalize_query
from f1_rag.retrieval.hybrid import HybridRetriever
from f1_rag.retrieval.keyword import KeywordRetriever
from f1_rag.retrieval.vector import VectorRetriever
from tests.conftest import make_chunk


def _deps():
    emb = HashingEmbedder(dim=128)
    chunks = [
        make_chunk("c1", "pit lane speed limit during the race is enforced", article_number="F4.1"),
        make_chunk("c2", "cost cap financial regulations for teams and reporting", article_number="D3.1", section=RegulationSection.D),
        make_chunk("c3", "fillet radius technical bodywork concave surface definition", article_number="C3.2.6", section=RegulationSection.C),
    ]
    store = NumpyVectorStore()
    store.build(chunks, emb.embed([c.text for c in chunks]))
    return RetrievalDeps(chunks=chunks, store=store, embedder=emb, top_k=3)


def test_normalize_query():
    assert normalize_query("  What   is  the\ncost cap? ") == "What is the cost cap?"


def test_min_max_normalize():
    assert min_max_normalize([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]
    assert min_max_normalize([2.0, 2.0]) == [0.0, 0.0]


def test_vector_retriever_scores():
    deps = _deps()
    cands = VectorRetriever(deps).retrieve("pit lane speed", k=3)
    assert cands[0].chunk.chunk_id == "c1"
    assert cands[0].similarity is not None and cands[0].vector_distance is not None


def test_keyword_retriever_scores():
    deps = _deps()
    cands = KeywordRetriever(deps).retrieve("cost cap reporting", k=3)
    assert cands[0].chunk.chunk_id == "c2"
    assert cands[0].keyword_score is not None


def test_hybrid_dedup_and_combined_score():
    deps = _deps()
    cands = HybridRetriever(deps).retrieve("cost cap financial", k=3)
    ids = [c.chunk.chunk_id for c in cands]
    assert len(ids) == len(set(ids))  # deduplicated by chunk_id
    assert cands[0].chunk.chunk_id == "c2"


def test_context_budget_and_dedup():
    dup_text = "identical chunk text about the pit lane speed limit"
    cands = [
        RetrievedCandidate(chunk=make_chunk("c1", dup_text), score=0.9),
        RetrievedCandidate(chunk=make_chunk("c2", dup_text), score=0.8),  # duplicate
        RetrievedCandidate(chunk=make_chunk("c3", "a" + " word" * 200), score=0.7),  # too big
    ]
    ctx = GreedyContextAssembler().assemble(cands, max_tokens=30)
    selected_ids = {c.chunk.chunk_id for c in ctx.selected}
    reasons = {d.chunk_id: d.reason for d in ctx.discarded}
    assert "c1" in selected_ids
    assert reasons.get("c2") == "duplicate_of_selected"
    assert reasons.get("c3") == "budget_exhausted"
    assert "[Art." in ctx.text  # citation header present
