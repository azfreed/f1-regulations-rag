"""Trace schema: a complete, inspectable record of one query.

This intentionally captures *everything* the requirements ask for, so nothing about
the RAG operation is hidden: the raw and normalized query, the embedding config,
every retrieved candidate with all its scores, the metadata filters, which chunks
were selected vs discarded (and why), the final prompt sent to the model, and the
citations in the answer.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..models import Citation, DiscardedChunk


class CandidateTrace(BaseModel):
    """One retrieved candidate as it appears in the trace."""

    rank: int | None = None
    chunk_id: str
    citation: str
    article_number: str
    section: str
    score: float
    similarity: float | None = None
    vector_distance: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    selected: bool = False
    text_preview: str = ""


class QueryTrace(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # --- query ---
    original_query: str = ""
    normalized_query: str = ""

    # --- embedding ---
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedding_config: dict = Field(default_factory=dict)

    # --- component selection ---
    retriever: str | None = None
    reranker: str | None = None
    index: str | None = None
    metadata_filters: dict = Field(default_factory=dict)
    top_k: int | None = None

    # --- retrieval / ranking ---
    candidates: list[CandidateTrace] = Field(default_factory=list)

    # --- context ---
    selected_chunk_ids: list[str] = Field(default_factory=list)
    discarded: list[DiscardedChunk] = Field(default_factory=list)
    context_token_estimate: int = 0

    # --- generation ---
    prompt_version: str | None = None
    generation_model: str | None = None
    final_prompt: str = ""
    answer_text: str = ""
    is_unanswerable: bool = False
    citations: list[Citation] = Field(default_factory=list)
