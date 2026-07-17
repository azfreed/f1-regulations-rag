"""Evaluation stage: measure retrieval + answer quality across configurations.

- :mod:`f1_rag.evaluation.dataset` : versioned eval cases (question + expectations).
- :mod:`f1_rag.evaluation.metrics`  : recall@k, MRR, article hit rate, citation
  accuracy, unsupported-claim rate, unanswerable handling.
- :mod:`f1_rag.evaluation.runner`   : run a dataset through a config and save a
  comparable experiment record.
"""

from __future__ import annotations

from .dataset import EvalCase, EvalDataset, load_dataset, seed_dataset
from .metrics import MetricSummary, compute_metrics
from .runner import run_evaluation

__all__ = [
    "EvalCase",
    "EvalDataset",
    "MetricSummary",
    "compute_metrics",
    "load_dataset",
    "run_evaluation",
    "seed_dataset",
]
