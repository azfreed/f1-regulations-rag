"""Shared pytest fixtures + helpers.

Tests use the deterministic ``hashing`` embedder and the ``numpy`` store so they
run fast, offline, and reproducibly (no model downloads, no network, no API keys).
"""

from __future__ import annotations

import pytest

from f1_rag.models import Article, Chunk, RegulationSection, SourceDocumentMeta


def make_chunk(
    chunk_id: str,
    text: str,
    article_number: str = "A1.1.1",
    section: RegulationSection = RegulationSection.A,
    page: int = 5,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        text=text,
        section=section,
        article_number=article_number,
        parent_article=article_number.rsplit(".", 1)[0] if "." in article_number else None,
        article_heading="Test heading",
        pdf_page_number=page,
        page_label=f"{section.value}{page}",
        source_filename=f"section-{section.value.lower()}-test.pdf",
        document_title=f"SECTION {section.value}: TEST",
        issue_number=1,
        publication_date="2026-06-25",
        token_estimate=max(1, len(text.split())),
    )


@pytest.fixture
def sample_meta() -> SourceDocumentMeta:
    return SourceDocumentMeta(
        source_filename="section-a-general.pdf",
        section=RegulationSection.A,
        document_title="SECTION A: GENERAL REGULATORY PROVISIONS",
        issue_number=3,
        publication_date="2026-06-25",
        wmsc_approval_date="2026-06-23",
        page_count=10,
    )


@pytest.fixture
def sample_articles() -> list[Article]:
    return [
        Article(
            section=RegulationSection.A,
            article_number="A1.1.1",
            parent_article="A1.1",
            heading="Overview",
            text=(
                "The FIA is responsible for the sporting organisation and regulation of "
                "the FIA Formula One World Championship, comprising the Grand Prix "
                "competitions on the International Sporting Calendar."
            ),
            pdf_page_number=5,
            page_label="A5",
            source_filename="section-a-general.pdf",
        ),
        Article(
            section=RegulationSection.A,
            article_number="A2.2",
            parent_article="A2",
            heading="Championship points system",
            text="Points are awarded to the first ten classified cars: 25, 18, 15 and so on.",
            pdf_page_number=9,
            page_label="A9",
            source_filename="section-a-general.pdf",
        ),
    ]
