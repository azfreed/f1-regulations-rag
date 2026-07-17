from __future__ import annotations

from f1_rag.evaluation.dataset import seed_dataset
from f1_rag.evaluation.metrics import CaseResult, compute_metrics


def test_seed_dataset_has_answerable_and_unanswerable():
    ds = seed_dataset()
    assert ds.version
    assert any(c.answerable for c in ds.cases)
    assert any(not c.answerable for c in ds.cases)


def test_recall_mrr_and_hit_rate_prefix_match():
    results = [
        # expected D3 hit by D3.1 at rank 1 -> recall 1.0, RR 1.0
        CaseResult(
            case_id="c1",
            answerable=True,
            expected_articles=["D3"],
            retrieved_articles=["D3.1", "A1.1"],
        ),
        # expected A2.2 appears at rank 2 -> RR 0.5
        CaseResult(
            case_id="c2",
            answerable=True,
            expected_articles=["A2.2"],
            retrieved_articles=["C3.1", "A2.2"],
        ),
    ]
    m = compute_metrics(results, k=2)
    assert m.recall_at_k == 1.0
    assert m.article_hit_rate == 1.0
    assert m.mrr == 0.75  # (1.0 + 0.5) / 2


def test_unanswerable_handling_and_unsupported_rate():
    results = [
        # answerable, cited article that is in context -> supported
        CaseResult(
            case_id="ok",
            answerable=True,
            expected_articles=["A2.2"],
            retrieved_articles=["A2.2"],
            context_articles=["A2.2"],
            answer_text="Points [Art. A2.2]",
            citation_articles=["A2.2"],
        ),
        # unanswerable correctly declined
        CaseResult(
            case_id="decline",
            answerable=False,
            answer_text="UNANSWERABLE: ...",
            is_unanswerable=True,
        ),
        # answerable but answered with no citations -> unsupported
        CaseResult(
            case_id="bad",
            answerable=True,
            expected_articles=["A3.1"],
            retrieved_articles=["A3.1"],
            context_articles=["A3.1"],
            answer_text="Some claim with no citation.",
            citation_articles=[],
        ),
    ]
    m = compute_metrics(results, k=3)
    assert m.unanswerable_handling == 1.0  # all 3 handled correctly
    # answered substantive cases = ok + bad; bad is unsupported -> 0.5
    assert m.unsupported_claim_rate == 0.5
    assert m.citation_accuracy == 1.0  # only 'ok' produced citations, and it matched
