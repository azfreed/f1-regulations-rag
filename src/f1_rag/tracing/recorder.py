"""TraceRecorder: accumulate a :class:`QueryTrace` as a query flows through stages.

The recorder is passed through the query pipeline (see ``cli.ask`` /
``pipeline.answer_query``). Each stage calls a small method to record its inputs
and outputs. Keeping this in one object means the pipeline code stays readable and
the trace is guaranteed to be consistent with what actually happened.
"""

from __future__ import annotations

from ..models import AssembledContext, Answer, RetrievedCandidate
from .models import CandidateTrace, QueryTrace


class TraceRecorder:
    def __init__(self) -> None:
        self.trace = QueryTrace()

    # --- query ---
    def record_query(self, original: str, normalized: str) -> None:
        self.trace.original_query = original
        self.trace.normalized_query = normalized

    # --- embedding / components ---
    def record_embedding(self, model: str | None, dim: int | None, config: dict) -> None:
        self.trace.embedding_model = model
        self.trace.embedding_dim = dim
        self.trace.embedding_config = config

    def record_components(
        self,
        retriever: str,
        reranker: str,
        index: str,
        metadata_filters: dict | None,
        top_k: int,
    ) -> None:
        self.trace.retriever = retriever
        self.trace.reranker = reranker
        self.trace.index = index
        self.trace.metadata_filters = metadata_filters or {}
        self.trace.top_k = top_k

    # --- candidates ---
    def record_candidates(
        self, candidates: list[RetrievedCandidate], selected_ids: set[str] | None = None
    ) -> None:
        selected_ids = selected_ids or set()
        traces: list[CandidateTrace] = []
        for cand in candidates:
            c = cand.chunk
            traces.append(
                CandidateTrace(
                    rank=cand.rank,
                    chunk_id=c.chunk_id,
                    citation=c.citation_label(),
                    article_number=c.article_number,
                    section=c.section.value,
                    score=cand.score,
                    similarity=cand.similarity,
                    vector_distance=cand.vector_distance,
                    keyword_score=cand.keyword_score,
                    rerank_score=cand.rerank_score,
                    selected=c.chunk_id in selected_ids,
                    text_preview=c.text[:200],
                )
            )
        self.trace.candidates = traces

    # --- context ---
    def record_context(self, context: AssembledContext) -> None:
        self.trace.selected_chunk_ids = [c.chunk.chunk_id for c in context.selected]
        self.trace.discarded = context.discarded
        self.trace.context_token_estimate = context.token_estimate
        # Mark selected flag on candidate traces (they were recorded before context).
        selected = set(self.trace.selected_chunk_ids)
        for ct in self.trace.candidates:
            ct.selected = ct.chunk_id in selected

    # --- generation ---
    def record_prompt(self, prompt_version: str, final_prompt: str) -> None:
        self.trace.prompt_version = prompt_version
        self.trace.final_prompt = final_prompt

    def record_answer(self, answer: Answer) -> None:
        self.trace.generation_model = answer.generation_model
        self.trace.answer_text = answer.text
        self.trace.is_unanswerable = answer.is_unanswerable
        self.trace.citations = answer.citations
