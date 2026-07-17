"""Visual asset detection + page rendering.

What it does
------------
For each PDF page it (1) renders a PNG page image, and (2) detects whether the page
is *likely* to contain a diagram, drawing, table, equation, or embedded image. It
records page image paths and, when available, extracted-image paths + bounding
boxes. This is kept completely separate from text extraction.

Why it exists
-------------
The regulations (especially Section C, Technical) rely on figures, CAD drawings,
and tables that text extraction cannot faithfully represent. Preserving page
images + detection flags now lets a future milestone build multimodal retrieval
that surfaces the original page, a figure, nearby text, and captions.

How detection works (heuristics, intentionally simple + visible)
----------------------------------------------------------------
- ``page.get_images()``     -> embedded raster images (bounding boxes via
  ``page.get_image_rects``). Presence => ``has_image`` / ``VisualKind.IMAGE``.
- ``page.get_drawings()``   -> vector paths. A high path count is a strong signal
  of a diagram/CAD drawing => ``has_diagram`` / ``has_drawing``.
- Table heuristic          -> many short, grid-aligned text blocks sharing x/y
  bands. This is approximate; see limitations.
- Equation heuristic       -> presence of math-ish glyphs / superscript density.

Limitations
-----------
- These are heuristics, not a layout model. Tables are the least reliable signal.
- Text-only extraction cannot reconstruct CAD drawings or complex tables; those
  pages are flagged so retrieval can point users at the page image instead.

Replaceable alternatives
-------------------------
- A layout model (e.g. a table/figure detector) could replace the heuristics while
  producing the same :class:`PageVisual` output.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF

from ..logging_utils import get_logger
from ..models import (
    ExtractedImage,
    PageVisual,
    RegulationSection,
    VisualDocument,
    VisualKind,
)

logger = get_logger(__name__)

# Tunable thresholds for the (deliberately simple) detectors.
DRAWING_PATH_THRESHOLD = 40  # vector paths above this => likely a diagram/drawing
TABLE_MIN_ALIGNED_ROWS = 4  # rows sharing column x-positions
# Minimum fraction of the page an embedded image must cover to count as "content".
# These PDFs carry a small FIA logo in the header on *every* page; without this
# filter has_image would be true everywhere and useless as a signal.
IMAGE_MIN_AREA_FRACTION = 0.03
_MATH_CHARS = set("=≤≥±×÷√∑∫∂∞≈≠→←°µΩπλ")


def _detect_tableish(blocks: list[tuple]) -> bool:
    """Very rough table heuristic: several rows whose blocks share x-start columns.

    We bucket block x0 positions and count how many distinct y-rows contain at least
    two column buckets. If several rows look column-aligned, we flag a table.
    """

    if len(blocks) < TABLE_MIN_ALIGNED_ROWS * 2:
        return False
    rows: dict[int, set[int]] = defaultdict(set)
    for b in blocks:
        x0, y0 = b[0], b[1]
        col = int(x0 // 40)  # 40pt-wide column buckets
        row = int(y0 // 12)  # 12pt-tall row buckets
        rows[row].add(col)
    multi_col_rows = sum(1 for cols in rows.values() if len(cols) >= 2)
    return multi_col_rows >= TABLE_MIN_ALIGNED_ROWS


def _detect_equation(text: str) -> bool:
    if not text:
        return False
    math_hits = sum(1 for ch in text if ch in _MATH_CHARS)
    # Superscript-ish patterns like x2, m3, 10^-3 also hint at equations.
    superscript = len(re.findall(r"\b\w\^?-?\d\b", text))
    return math_hits >= 3 or (math_hits >= 1 and superscript >= 3)


def extract_visuals(
    pdf_path: str | Path,
    section: RegulationSection,
    output_dir: str | Path,
    dpi: int = 150,
) -> VisualDocument:
    """Render page images and collect per-page visual metadata."""

    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir) / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    page_visuals: list[PageVisual] = []
    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            page_no = i + 1
            img_path = out_dir / f"page-{page_no:03d}.png"
            # Render the page to a raster image (idempotent: overwrite is fine).
            pix = page.get_pixmap(matrix=matrix)
            pix.save(img_path)

            # --- embedded raster images + bounding boxes ---
            page_area = float(page.rect.width * page.rect.height) or 1.0
            extracted: list[ExtractedImage] = []
            has_substantial_image = False
            for img in page.get_images(full=True):
                xref = img[0]
                bbox = None
                try:
                    rects = page.get_image_rects(xref)
                    if rects:
                        r = rects[0]
                        bbox = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                        area_frac = (r.width * r.height) / page_area
                        if area_frac >= IMAGE_MIN_AREA_FRACTION:
                            has_substantial_image = True
                except Exception:  # noqa: BLE001 - bbox is best-effort
                    bbox = None
                extracted.append(ExtractedImage(image_path=str(img_path), bbox=bbox, xref=xref))

            # --- vector drawings ---
            try:
                drawing_count = len(page.get_drawings())
            except Exception:  # noqa: BLE001
                drawing_count = 0

            blocks = page.get_text("blocks")
            text = page.get_text("text")

            # Only a substantial embedded image (not the recurring header logo)
            # counts as a visual-content signal.
            has_image = has_substantial_image
            has_drawing = drawing_count >= DRAWING_PATH_THRESHOLD
            has_diagram = has_drawing  # a dense vector page is our best diagram proxy
            has_table = _detect_tableish(blocks)
            has_equation = _detect_equation(text)

            kinds: list[VisualKind] = []
            if has_image:
                kinds.append(VisualKind.IMAGE)
            if has_diagram:
                kinds.append(VisualKind.DIAGRAM)
            if has_drawing:
                kinds.append(VisualKind.DRAWING)
            if has_table:
                kinds.append(VisualKind.TABLE)
            if has_equation:
                kinds.append(VisualKind.EQUATION)

            page_visuals.append(
                PageVisual(
                    pdf_page_number=page_no,
                    page_image_path=str(img_path),
                    has_diagram=has_diagram,
                    has_drawing=has_drawing,
                    has_table=has_table,
                    has_equation=has_equation,
                    has_image=has_image,
                    drawing_path_count=drawing_count,
                    image_count=len(extracted),
                    extracted_images=extracted,
                    kinds=kinds,
                )
            )

    likely = sum(1 for p in page_visuals if p.is_likely_visual)
    logger.info(
        "rendered %d pages for %s (%d flagged as likely-visual)",
        len(page_visuals),
        pdf_path.name,
        likely,
    )
    return VisualDocument(
        source_filename=pdf_path.name, section=section, pages=page_visuals
    )
