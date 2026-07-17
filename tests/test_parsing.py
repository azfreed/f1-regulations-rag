from __future__ import annotations

from f1_rag.models import CleanPage, RegulationSection, SourceDocumentMeta
from f1_rag.parsing.headings import looks_like_heading, parent_of
from f1_rag.parsing.regulations import parse_articles


def test_parent_of():
    assert parent_of("A1") is None
    assert parent_of("A1.2") == "A1"
    assert parent_of("A1.2.2") == "A1.2"


def test_looks_like_heading():
    assert looks_like_heading("Overview") is True
    assert looks_like_heading("Applicable regulations") is True
    assert looks_like_heading("The FIA is responsible for the Championship.") is False
    assert looks_like_heading('The "FIA F1 Regulations" comprise the following:') is False
    assert looks_like_heading("17") is False  # a page number


def _meta() -> SourceDocumentMeta:
    return SourceDocumentMeta(
        source_filename="section-a-general.pdf",
        section=RegulationSection.A,
        document_title="SECTION A",
        page_count=1,
    )


def test_parse_split_number_heading_lines():
    # Mimics PyMuPDF text-mode output: number and heading/text on separate lines.
    page = CleanPage(
        pdf_page_number=5,
        page_label="A5",
        is_toc=False,
        text="\n".join(
            [
                "ARTICLE A1: GENERAL PRINCIPLES",
                "Advisory Committee: RGAC",
                "A1.1",
                "Overview",
                "A1.1.1",
                "The FIA is responsible for the sporting organisation of the Championship.",
                "A1.1.2",
                "The Championship is the exclusive property of the FIA.",
            ]
        ),
    )
    articles = parse_articles(_meta(), [page])
    by_num = {a.article_number: a for a in articles}

    assert "A1" in by_num and by_num["A1"].heading == "GENERAL PRINCIPLES"
    assert "A1.1" in by_num and by_num["A1.1"].heading == "Overview"
    # Provisions capture prose and inherit the subarticle heading; annotation skipped.
    assert "responsible" in by_num["A1.1.1"].text
    assert "Advisory Committee" not in by_num["A1.1.1"].text
    assert by_num["A1.1.1"].parent_article == "A1.1"


def test_toc_pages_excluded_from_parsing():
    toc = CleanPage(pdf_page_number=2, is_toc=True, text="A1.1\nOverview\n5")
    body = CleanPage(
        pdf_page_number=5,
        is_toc=False,
        text="A2.2\nPoints are awarded to the first ten classified cars.",
    )
    articles = parse_articles(_meta(), [toc, body])
    assert [a.article_number for a in articles] == ["A2.2"]
