"""Evaluation runner: run a dataset through a config and save a comparable record.

What it does
------------
For each case, runs the query pipeline (retrieval + context, and optionally
generation), builds a :class:`CaseResult`, computes the :class:`MetricSummary`, and
writes an experiment record to ``experiments/runs/<timestamp>.json`` containing the
full configuration (corpus version, chunking config, embedding model, retriever,
reranker, generation model, prompt version) alongside the metrics.

Why it exists
-------------
This is what makes configurations *comparable*: every run records the same shape of
config + metrics, so you can diff ``vector`` vs ``hybrid`` or ``regulation`` vs
``fixed_window`` directly.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings
from ..generation.prompts import DEFAULT_PROMPT_VERSION
from ..logging_utils import get_logger
from ..pipeline import QueryEngine
from .dataset import EvalDataset
from .metrics import CaseResult, MetricSummary, compute_metrics

logger = get_logger(__name__)


def run_evaluation(
    settings: Settings,
    dataset: EvalDataset,
    chunker: str | None = None,
    embedder: str | None = None,
    index: str | None = None,
    retriever: str | None = None,
    reranker: str | None = None,
    k: int | None = None,
    generate: bool = False,
) -> tuple[MetricSummary, Path]:
    """Run the dataset and persist an experiment record. Returns (metrics, path)."""

    engine = QueryEngine.load(settings, chunker=chunker, embedder=embedder, index=index)
    k = k or settings.top_k

    results: list[CaseResult] = []
    for case in dataset.cases:
        answer, trace = engine.answer(
            case.question,
            retriever=retriever,
            reranker=reranker,
            k=k,
            generate=generate,
        )
        retrieved_articles = [c.article_number for c in trace.candidates]
        context_articles = [
            c.article_number for c in trace.candidates if c.selected
        ]
        results.append(
            CaseResult(
                case_id=case.id,
                answerable=case.answerable,
                expected_articles=case.expected_articles,
                expected_answer_terms=case.expected_answer_terms,
                retrieved_articles=retrieved_articles,
                context_articles=context_articles,
                answer_text=answer.text if answer else None,
                is_unanswerable=answer.is_unanswerable if answer else False,
                citation_articles=[c.article_number for c in answer.citations] if answer else [],
            )
        )

    metrics = compute_metrics(results, k=k)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset.version,
        "corpus_version": engine.manifest.get("corpus_hash"),
        "config": {
            "chunker": chunker or settings.chunker,
            "chunk_max_tokens": engine.manifest.get("chunk_max_tokens"),
            "chunk_overlap_tokens": engine.manifest.get("chunk_overlap_tokens"),
            "embedding_model": engine.manifest.get("embedding_model"),
            "embedder": embedder or settings.embedder,
            "index": index or settings.index,
            "retriever": retriever or settings.retriever,
            "reranker": reranker or settings.reranker,
            "generation_model": settings.generation_model if generate else None,
            "prompt_version": DEFAULT_PROMPT_VERSION if generate else None,
            "k": k,
        },
        "metrics": metrics.model_dump(),
        "per_case": [r.model_dump() for r in results],
    }

    settings.ensure_dirs()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = settings.runs_dir / f"{ts}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("evaluation saved to %s", out_path)
    return metrics, out_path
