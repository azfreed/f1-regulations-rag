# Visual content: current and planned handling

The FIA regulations - especially **Section C (Technical)** - depend heavily on
figures, CAD drawings, dimensioned diagrams, and tables. Text extraction alone
cannot faithfully represent these, so this project captures visual metadata from
the very first milestone even though multimodal *retrieval* is deferred.

## What is implemented now (`extraction/visuals.py`)

For every page we:

1. **Render a page image** (`data/visual/<section>/page-NNN.png`, 150 dpi).
2. **Detect likely visual content** with simple, transparent heuristics:
   - **Embedded images** via `page.get_images()`, with bounding boxes from
     `page.get_image_rects()`.
   - **Vector drawings / diagrams** via `page.get_drawings()` path counts (dense
     vector pages ≈ CAD drawings).
   - **Tables** via a block-alignment heuristic (several rows sharing column
     x-positions).
   - **Equations** via math-glyph / superscript density.
3. **Store the metadata** to `data/visual/<section>.visual.jsonl` as `PageVisual`
   records, and record the page image path on every `Chunk`.

This stage is intentionally **separate** from text extraction and does not affect
the text corpus.

## The data model is ready for multimodal surfacing

`PageVisual`, `ExtractedImage`, and the `page_image_path` on each `Chunk` are
designed so a future result can surface:

- the **original PDF page** (we have the page number and label),
- a **page image** (`page_image_path`),
- an **extracted figure/diagram** (`ExtractedImage.image_path` + `bbox`),
- **nearby text** (the chunk text and adjacent chunks on the same page),
- a **caption** (a future enhancement: detect caption lines near a figure).

## Limitations of text-only extraction

- **CAD drawings** (much of Section C) carry their meaning in geometry, not text;
  the extracted text near them is often just labels and callouts. Retrieval should
  point users at the **page image**, which is why we render every page now.
- **Tables** lose their row/column structure when flattened to text; values can be
  misattributed. The table heuristic flags these pages so they can be treated
  cautiously or rendered.
- **Equations** become lossy inline text; the equation flag marks pages where the
  rendered image is the source of truth.

## Planned (not in this milestone)

- Figure/caption pairing and per-figure crops from bounding boxes.
- A multimodal embedding path so a diagram-heavy query can retrieve the page image
  alongside text.
- A table extractor that preserves structure for financial (Sections D/E) tables.

These are deliberately out of scope for the first milestone; the metadata captured
now is what makes them straightforward to add later.
