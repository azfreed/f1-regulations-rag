from __future__ import annotations

import pytest

from f1_rag.errors import GenerationError
from f1_rag.generation.base import extract_citations
from f1_rag.generation.prompts import get_prompt
from f1_rag.models import AssembledContext, Answer, RetrievedCandidate
from f1_rag.tracing.recorder import TraceRecorder
from tests.conftest import make_chunk


def _context():
    cands = [
        RetrievedCandidate(chunk=make_chunk("c1", "points text", article_number="A2.2"), score=0.9, similarity=0.9, vector_distance=0.1, rank=0),
        RetrievedCandidate(chunk=make_chunk("c2", "entry text", article_number="A3.1"), score=0.8, similarity=0.8, vector_distance=0.2, rank=1),
    ]
    return AssembledContext(text="[Art. A2.2 ...]\npoints text", selected=cands, token_estimate=10)


def test_extract_citations_only_used_and_present():
    ctx = _context()
    answer = "Points are awarded per [Art. A2.2 (Section A, p.A9)]."
    cits = extract_citations(answer, ctx)
    assert [c.article_number for c in cits] == ["A2.2"]  # A3.1 in context but not cited


def test_prompt_versioned_and_grounding_policy():
    p = get_prompt()
    assert p.version == "v1"
    assert "ONLY" in p.system
    assert "UNANSWERABLE" in p.system


def test_anthropic_generator_refuses_without_context():
    from f1_rag.generation.anthropic_client import AnthropicGenerator

    gen = AnthropicGenerator(api_key="dummy-key")  # no network call happens
    empty = AssembledContext(text="", selected=[], token_estimate=0)
    with pytest.raises(GenerationError):
        gen.generate("anything", empty)


def test_anthropic_generator_requires_api_key():
    from f1_rag.generation.anthropic_client import AnthropicGenerator

    with pytest.raises(GenerationError):
        AnthropicGenerator(api_key=None)


def test_trace_recorder_populates_trace():
    rec = TraceRecorder()
    rec.record_query("Q?", "q")
    rec.record_embedding("hashing", 256, {"batch_size": 32})
    rec.record_components("hybrid", "none", "numpy", {"section": "A"}, 8)
    ctx = _context()
    rec.record_candidates(ctx.selected)
    rec.record_context(ctx)
    rec.record_prompt("v1", "the prompt")
    rec.record_answer(Answer(question="Q?", text="answer [Art. A2.2 ...]", generation_model="claude"))

    t = rec.trace
    assert t.normalized_query == "q"
    assert t.embedding_model == "hashing"
    assert t.retriever == "hybrid"
    assert t.metadata_filters == {"section": "A"}
    assert len(t.candidates) == 2
    assert all(c.selected for c in t.candidates)  # both selected in context
    assert t.final_prompt == "the prompt"
    assert t.generation_model == "claude"
