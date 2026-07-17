from __future__ import annotations

from f1_rag.extraction.cleanup import clean_document
from f1_rag.models import RawDocument, RawPage, SourceDocumentMeta, RegulationSection, TextBlock


def _page(n: int, text: str, label: str | None = None) -> RawPage:
    # Put header/footer text in the top/bottom bands so frequency detection sees them.
    blocks = [
        TextBlock(text="SECTION A: GENERAL REGULATORY PROVISIONS", x0=0, y0=5, x1=400, y1=20),
        TextBlock(text=text, x0=0, y0=300, x1=400, y1=600),
        TextBlock(text="©2026 Fédération Internationale de l'Automobile", x0=0, y0=780, x1=400, y1=795),
    ]
    header = "SECTION A: GENERAL REGULATORY PROVISIONS"
    footer = "©2026 Fédération Internationale de l'Automobile"
    full = f"{header}\n{label or ''}\n{text}\n{footer}"
    return RawPage(pdf_page_number=n, page_label=label, raw_text=full, blocks=blocks, width=400, height=800)


def _doc(pages: list[RawPage]) -> RawDocument:
    meta = SourceDocumentMeta(
        source_filename="section-a-general.pdf",
        section=RegulationSection.A,
        document_title="SECTION A",
        page_count=len(pages),
    )
    return RawDocument(meta=meta, pages=pages)


def test_headers_and_footers_removed():
    pages = [_page(i, f"Body text for page {i}", label=f"A{i}") for i in range(1, 6)]
    clean = clean_document(_doc(pages))
    for cp in clean:
        assert "SECTION A: GENERAL REGULATORY PROVISIONS" not in cp.text
        assert "Fédération" not in cp.text
        assert "Body text" in cp.text


def test_toc_flagged_only_in_front_matter():
    # Front matter: a CONTENTS page and a page full of bare page-number ints.
    contents = _page(1, "CONTENTS\nARTICLE A1: GENERAL PRINCIPLES", label="A1")
    toc = _page(2, "A1.1\nOverview\n5\nA1.2\nApplicable regulations\n7\n9\n11\n13", label="A2")
    body_start = _page(3, "A1.1.1\nThe FIA is responsible for the Championship.", label="A3")
    # A deep-body page with a numeric table must NOT be flagged as TOC.
    table = _page(4, "A2.2\n25\n18\n15\n12\n10\n8\nPoints awarded to classified cars.", label="A4")
    clean = clean_document(_doc([contents, toc, body_start, table]))
    flags = {cp.pdf_page_number: cp.is_toc for cp in clean}
    assert flags[1] is True
    assert flags[2] is True
    assert flags[3] is False
    assert flags[4] is False  # numeric table deep in body, not a TOC
