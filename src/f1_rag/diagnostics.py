"""Per-PDF diagnostic report (milestone 1 deliverable).

For each PDF this produces a report showing: detected document metadata, the first
detected articles, pages likely to contain visual content, and extraction
warnings. Written as both Markdown (human-readable) and JSON (inspectable) under
``experiments/diagnostics/``.

This is deliberately its own module so the milestone's "inspect before indexing"
goal is a single, cheap command that touches only extraction + parsing + visuals.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import Settings
from .extraction.cleanup import clean_document
from .extraction.text import extract_document
from .extraction.visuals import extract_visuals
from .logging_utils import get_logger
from .models import write_jsonl
from .parsing.regulations import parse_articles

logger = get_logger(__name__)


@dataclass
class DiagnosticReport:
    source_filename: str
    section: str
    document_title: str
    issue_number: int | None
    publication_date: str | None
    wmsc_approval_date: str | None
    page_count: int
    n_articles: int
    n_toc_pages: int
    first_articles: list[dict] = field(default_factory=list)
    likely_visual_pages: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def diagnose_pdf(
    settings: Settings,
    pdf_path: Path,
    render_visuals: bool = True,
    max_articles: int = 15,
) -> DiagnosticReport:
    raw = extract_document(pdf_path)
    meta = raw.meta

    warnings: list[str] = []
    if meta.issue_number is None:
        warnings.append("issue number not detected on page 1")
    if meta.publication_date is None:
        warnings.append("publication date not detected on page 1")
    if meta.wmsc_approval_date is None:
        warnings.append("WMSC approval date not detected on page 1")

    clean_pages = clean_document(raw)
    n_toc = sum(1 for p in clean_pages if p.is_toc)
    if n_toc == 0:
        warnings.append("no table-of-contents pages detected (unexpected for these PDFs)")

    articles = parse_articles(meta, clean_pages)
    if not articles:
        warnings.append("no articles parsed - heading detection may have failed")
    pages_with_labels = sum(1 for p in raw.pages if p.page_label)
    if pages_with_labels < raw.meta.page_count * 0.5:
        warnings.append(
            f"only {pages_with_labels}/{raw.meta.page_count} pages had a detectable page label"
        )

    first_articles = [
        {
            "article_number": a.article_number,
            "parent_article": a.parent_article,
            "heading": a.heading,
            "pdf_page_number": a.pdf_page_number,
            "page_label": a.page_label,
            "text_preview": a.text[:120],
        }
        for a in articles[:max_articles]
    ]

    likely_visual: list[dict] = []
    if render_visuals:
        visual = extract_visuals(pdf_path, meta.section, settings.visual_dir)
        write_jsonl(settings.visual_dir / f"{pdf_path.stem}.visual.jsonl", visual.pages)
        for p in visual.pages:
            if p.is_likely_visual:
                likely_visual.append(
                    {
                        "pdf_page_number": p.pdf_page_number,
                        "kinds": [k.value for k in p.kinds],
                        "drawing_path_count": p.drawing_path_count,
                        "image_count": p.image_count,
                        "page_image_path": p.page_image_path,
                    }
                )

    return DiagnosticReport(
        source_filename=meta.source_filename,
        section=meta.section.value,
        document_title=meta.document_title,
        issue_number=meta.issue_number,
        publication_date=meta.publication_date,
        wmsc_approval_date=meta.wmsc_approval_date,
        page_count=meta.page_count,
        n_articles=len(articles),
        n_toc_pages=n_toc,
        first_articles=first_articles,
        likely_visual_pages=likely_visual,
        warnings=warnings,
    )


def render_report_markdown(report: DiagnosticReport) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# Diagnostic report: {report.source_filename}")
    a("")
    a("## Detected metadata")
    a("")
    a(f"- Section: {report.section}")
    a(f"- Title: {report.document_title}")
    a(f"- Issue number: {report.issue_number}")
    a(f"- Publication date: {report.publication_date}")
    a(f"- WMSC approval date: {report.wmsc_approval_date}")
    a(f"- Page count: {report.page_count}")
    a(f"- Parsed article/subarticle records: {report.n_articles}")
    a(f"- TOC pages flagged: {report.n_toc_pages}")
    a("")
    a("## First detected articles")
    a("")
    if report.first_articles:
        a("| Article | Parent | Heading | Page | Label |")
        a("| --- | --- | --- | --- | --- |")
        for fa in report.first_articles:
            heading = (fa["heading"] or "").replace("|", "\\|")
            a(
                f"| {fa['article_number']} | {fa['parent_article'] or ''} | "
                f"{heading} | {fa['pdf_page_number']} | {fa['page_label'] or ''} |"
            )
    else:
        a("_none detected_")
    a("")
    a("## Likely visual pages")
    a("")
    if report.likely_visual_pages:
        a(f"{len(report.likely_visual_pages)} pages flagged. First 25:")
        a("")
        a("| PDF page | Kinds | Drawing paths | Images |")
        a("| --- | --- | --- | --- |")
        for v in report.likely_visual_pages[:25]:
            a(
                f"| {v['pdf_page_number']} | {', '.join(v['kinds'])} | "
                f"{v['drawing_path_count']} | {v['image_count']} |"
            )
    else:
        a("_none detected_")
    a("")
    a("## Extraction warnings")
    a("")
    if report.warnings:
        for w in report.warnings:
            a(f"- {w}")
    else:
        a("- none")
    a("")
    return "\n".join(lines)


def run_diagnostics(
    settings: Settings, pdf_filenames: list[str] | None = None, render_visuals: bool = True
) -> list[Path]:
    settings.ensure_dirs()
    pdfs = (
        [settings.raw_dir / fn for fn in pdf_filenames]
        if pdf_filenames
        else sorted(settings.raw_dir.glob("*.pdf"))
    )
    written: list[Path] = []
    for pdf in pdfs:
        logger.info("diagnosing %s", pdf.name)
        report = diagnose_pdf(settings, pdf, render_visuals=render_visuals)
        md_path = settings.diagnostics_dir / f"{pdf.stem}.md"
        json_path = settings.diagnostics_dir / f"{pdf.stem}.json"
        md_path.write_text(render_report_markdown(report), encoding="utf-8")
        json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        written.extend([md_path, json_path])
    return written
