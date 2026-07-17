"""Render a :class:`QueryTrace` as human-readable text and as JSON.

The human-readable form is meant to be scanned in a terminal; the JSON form is the
inspectable artifact saved under ``experiments/traces/`` for later analysis.
"""

from __future__ import annotations

from pathlib import Path

from .models import QueryTrace


def render_trace_text(trace: QueryTrace) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("QUERY TRACE")
    add("=" * 78)
    add(f"original query   : {trace.original_query}")
    add(f"normalized query : {trace.normalized_query}")
    add(f"embedding model  : {trace.embedding_model} (dim={trace.embedding_dim})")
    add(f"embedding config : {trace.embedding_config}")
    add(f"index / retriever: {trace.index} / {trace.retriever}   reranker: {trace.reranker}")
    add(f"metadata filters : {trace.metadata_filters or '(none)'}")
    add(f"top_k            : {trace.top_k}")
    add("")
    add(f"CANDIDATES ({len(trace.candidates)}) - '*' = selected for context")
    add("-" * 78)
    add(
        f"{'sel':<3} {'rank':<4} {'article':<10} {'score':>7} {'cos':>7} "
        f"{'dist':>7} {'bm25':>8} {'rerank':>7}  citation"
    )
    for c in trace.candidates:
        sel = "*" if c.selected else " "
        add(
            f"{sel:<3} {str(c.rank):<4} {c.article_number:<10} "
            f"{c.score:>7.3f} "
            f"{('%.3f' % c.similarity) if c.similarity is not None else '   -  ':>7} "
            f"{('%.3f' % c.vector_distance) if c.vector_distance is not None else '   -  ':>7} "
            f"{('%.3f' % c.keyword_score) if c.keyword_score is not None else '    -   ':>8} "
            f"{('%.3f' % c.rerank_score) if c.rerank_score is not None else '   -  ':>7}  "
            f"{c.citation}"
        )

    if trace.discarded:
        add("")
        add(f"DISCARDED ({len(trace.discarded)})")
        add("-" * 78)
        for d in trace.discarded:
            add(f"  - {d.citation}: {d.reason}")

    add("")
    add(f"CONTEXT: {len(trace.selected_chunk_ids)} chunks, "
        f"~{trace.context_token_estimate} tokens")
    add("")
    add(f"PROMPT (version {trace.prompt_version}, model {trace.generation_model})")
    add("-" * 78)
    if trace.final_prompt:
        preview = trace.final_prompt
        add(preview if len(preview) <= 4000 else preview[:4000] + "\n... [truncated]")
    else:
        add("(no prompt - generation not run)")

    add("")
    add("ANSWER" + ("  [UNANSWERABLE]" if trace.is_unanswerable else ""))
    add("-" * 78)
    add(trace.answer_text or "(no answer generated)")
    if trace.citations:
        add("")
        add("CITATIONS")
        for cit in trace.citations:
            add(f"  - {cit.label}  [{cit.source_filename}]")
    add("=" * 78)
    return "\n".join(lines)


def write_trace_json(trace: QueryTrace, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
    return path
