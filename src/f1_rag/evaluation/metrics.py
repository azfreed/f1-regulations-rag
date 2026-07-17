"""Retrieval + answer-quality metrics.

All metrics operate on :class:`CaseResult` records so they are decoupled from how
retrieval/generation was run. Article matching uses *prefix* semantics: an expected
article ``"D3"`` is considered hit by any retrieved article ``"D3"`` or ``"D3.x"``,
because expectations are often stated at a coarser granularity than chunks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CaseResult(BaseModel):
    """Per-case outcome captured by the runner, consumed by the metrics."""

    case_id: str
    answerable: bool
    expected_articles: list[str] = Field(default_factory=list)
    expected_answer_terms: list[str] = Field(default_factory=list)
    retrieved_articles: list[str] = Field(default_factory=list)  # ranked, top-k
    context_articles: list[str] = Field(default_factory=list)  # in final context
    answer_text: str | None = None
    is_unanswerable: bool = False
    citation_articles: list[str] = Field(default_factory=list)


class MetricSummary(BaseModel):
    n_cases: int
    n_answerable: int
    recall_at_k: float
    mrr: float
    article_hit_rate: float
    citation_accuracy: float
    unsupported_claim_rate: float
    unanswerable_handling: float
    answer_term_coverage: float
    k: int


def _article_matches(expected: str, retrieved: str) -> bool:
    return retrieved == expected or retrieved.startswith(expected + ".") or retrieved.startswith(expected)


def _hit(expected_articles: list[str], retrieved: list[str]) -> bool:
    return any(_article_matches(e, r) for e in expected_articles for r in retrieved)


def _recall(expected_articles: list[str], retrieved: list[str]) -> float:
    if not expected_articles:
        return 1.0
    found = sum(1 for e in expected_articles if any(_article_matches(e, r) for r in retrieved))
    return found / len(expected_articles)


def _reciprocal_rank(expected_articles: list[str], retrieved: list[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if any(_article_matches(e, r) for e in expected_articles):
            return 1.0 / rank
    return 0.0


def compute_metrics(results: list[CaseResult], k: int) -> MetricSummary:
    n = len(results)
    answerable = [r for r in results if r.answerable]
    n_ans = len(answerable)

    # --- retrieval metrics (over answerable cases with expected articles) ---
    scored = [r for r in answerable if r.expected_articles]
    recall = sum(_recall(r.expected_articles, r.retrieved_articles) for r in scored) / len(scored) if scored else 0.0
    mrr = sum(_reciprocal_rank(r.expected_articles, r.retrieved_articles) for r in scored) / len(scored) if scored else 0.0
    hit_rate = sum(1 for r in scored if _hit(r.expected_articles, r.retrieved_articles)) / len(scored) if scored else 0.0

    # --- citation accuracy: of cases that produced citations, fraction where the
    #     citations match the expected articles. ---
    cited = [r for r in scored if r.citation_articles]
    citation_acc = (
        sum(1 for r in cited if _hit(r.expected_articles, r.citation_articles)) / len(cited)
        if cited
        else 0.0
    )

    # --- unsupported-claim rate: answered (non-unanswerable) cases whose citations
    #     are NOT all present in the retrieved context, or that cite nothing. ---
    answered = [r for r in results if r.answer_text and not r.is_unanswerable]
    def _unsupported(r: CaseResult) -> bool:
        if not r.citation_articles:
            return True  # a substantive answer with no citation is unsupported
        ctx = set(r.context_articles)
        return not all(c in ctx for c in r.citation_articles)
    unsupported_rate = (sum(1 for r in answered if _unsupported(r)) / len(answered)) if answered else 0.0

    # --- unanswerable handling: unanswerable cases correctly declined, plus
    #     answerable cases NOT wrongly declined. ---
    correct = 0
    for r in results:
        if r.answerable and not r.is_unanswerable:
            correct += 1
        elif not r.answerable and r.is_unanswerable:
            correct += 1
    unanswerable_handling = correct / n if n else 0.0

    # --- answer term coverage (light lexical check on answerable cases) ---
    term_cases = [r for r in answered if r.expected_answer_terms]
    def _coverage(r: CaseResult) -> float:
        text = (r.answer_text or "").lower()
        hits = sum(1 for t in r.expected_answer_terms if t.lower() in text)
        return hits / len(r.expected_answer_terms)
    term_cov = sum(_coverage(r) for r in term_cases) / len(term_cases) if term_cases else 0.0

    return MetricSummary(
        n_cases=n,
        n_answerable=n_ans,
        recall_at_k=round(recall, 4),
        mrr=round(mrr, 4),
        article_hit_rate=round(hit_rate, 4),
        citation_accuracy=round(citation_acc, 4),
        unsupported_claim_rate=round(unsupported_rate, 4),
        unanswerable_handling=round(unanswerable_handling, 4),
        answer_term_coverage=round(term_cov, 4),
        k=k,
    )
